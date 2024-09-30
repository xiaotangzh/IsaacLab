# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import enum
import numpy as np
import sys
import torch
import traceback
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import carb
import omni.kit.app
import omni.kit.loop._loop as omni_loop
import omni.physics.tensors
import omni.physx
import omni.timeline
import omni.usd
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

from .simulation_cfg import SimulationCfg
from .spawners import DomeLightCfg, GroundPlaneCfg
from .utils import bind_physics_material


class SimulationContext:
    """A class to control simulation-related events such as physics stepping and rendering.

    The simulation context helps control various simulation aspects. This includes:

    * configure the simulator with different settings such as the physics time-step, the number of physics substeps,
      and the physics solver parameters (for more information, see :class:`omni.isaac.lab.sim.SimulationCfg`)
    * playing, pausing, stepping and stopping the simulation
    * adding and removing callbacks to different simulation events such as physics stepping, rendering, etc.

    This class inherits from the :class:`omni.isaac.core.simulation_context.SimulationContext` class and
    adds additional functionalities such as setting up the simulation context with a configuration object,
    exposing other commonly used simulator-related functions, and performing version checks of Isaac Sim
    to ensure compatibility between releases.

    The simulation context is a singleton object. This means that there can only be one instance
    of the simulation context at any given time. This is enforced by the parent class. Therefore, it is
    not possible to create multiple instances of the simulation context. Instead, the simulation context
    can be accessed using the ``instance()`` method.

    .. attention::
        Since we only support the `PyTorch <https://pytorch.org/>`_ backend for simulation, the
        simulation context is configured to use the ``torch`` backend by default. This means that
        all the data structures used in the simulation are ``torch.Tensor`` objects.

    The simulation context can only be used in standalone python scripts. In this mode, the user has full
    control over the simulation and can trigger stepping events synchronously (i.e. as a blocking call).
    The user has to manually call :meth:`step` step the physics simulation and :meth:`render` to render the
    scene.
    """

    class RenderMode(enum.IntEnum):
        """Different rendering modes for the simulation.

        Render modes correspond to how the viewport and other UI elements (such as listeners to keyboard or mouse
        events) are updated. There are three main components that can be updated when the simulation is rendered:

        1. **UI elements and other extensions**: These are UI elements (such as buttons, sliders, etc.) and other
           extensions that are running in the background that need to be updated when the simulation is running.
        2. **Cameras**: These are typically based on Hydra textures and are used to render the scene from different
           viewpoints. They can be attached to a viewport or be used independently to render the scene.
        3. **Viewports**: These are windows where you can see the rendered scene.

        Updating each of the above components has a different overhead. For example, updating the viewports is
        computationally expensive compared to updating the UI elements. Therefore, it is useful to be able to
        control what is updated when the simulation is rendered. This is where the render mode comes in. There are
        four different render modes:

        * :attr:`NO_GUI_OR_RENDERING`: The simulation is running without a GUI and off-screen rendering flag is disabled,
          so none of the above are updated.
        * :attr:`NO_RENDERING`: No rendering, where only 1 is updated at a lower rate.
        * :attr:`PARTIAL_RENDERING`: Partial rendering, where only 1 and 2 are updated.
        * :attr:`FULL_RENDERING`: Full rendering, where everything (1, 2, 3) is updated.

        .. _Viewports: https://docs.omniverse.nvidia.com/extensions/latest/ext_viewport.html
        """

        NO_GUI_OR_RENDERING = -1
        """The simulation is running without a GUI and off-screen rendering is disabled."""
        NO_RENDERING = 0
        """No rendering, where only other UI elements are updated at a lower rate."""
        PARTIAL_RENDERING = 1
        """Partial rendering, where the simulation cameras and UI elements are updated."""
        FULL_RENDERING = 2
        """Full rendering, where all the simulation viewports, cameras and UI elements are updated."""

    _instance: SimulationContext | None = None
    """The singleton instance of the simulation context."""

    _is_initialized: bool = False
    """Whether the simulation context is initialized."""

    def __init__(self, cfg: SimulationCfg | None = None):
        """Creates a simulation context to control the simulator.

        Args:
            cfg: The configuration of the simulation. Defaults to None,
                in which case the default configuration is used.
        """
        # check if the instance is already initialized
        if SimulationContext._is_initialized:
            carb.log_warn("Simulation context is already initialized. Returning the existing instance.")
            return
        # set the initialized flag
        SimulationContext._is_initialized = True

        # store input
        if cfg is None:
            cfg = SimulationCfg()
        self.cfg = cfg
        # check that simulation is running
        if omni.usd.get_context().get_stage() is None:
            raise RuntimeError("The stage has not been created. Did you run the simulator?")

        # set flags for simulator
        # acquire settings interface
        self._settings_iface = carb.settings.get_settings()
        # enable hydra scene-graph instancing
        # note: this allows rendering of instanceable assets on the GUI
        self._settings_iface.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
        # note: we read this once since it is not expected to change during runtime
        # read flag for whether a local GUI is enabled
        self._local_gui = self._settings_iface.get("/app/window/enabled")
        # read flag for whether livestreaming GUI is enabled
        self._livestream_gui = self._settings_iface.get("/app/livestream/enabled")

        # read flag for whether the Isaac Lab viewport capture pipeline will be used,
        # casting None to False if the flag doesn't exist
        # this flag is set from the AppLauncher class
        self._offscreen_render = bool(self._settings_iface.get("/isaaclab/render/offscreen"))
        # read flag for whether the default viewport should be enabled
        self._render_viewport = bool(self._settings_iface.get("/isaaclab/render/active_viewport"))
        # flag for whether any GUI will be rendered (local, livestreamed or viewport)
        self._has_gui = self._local_gui or self._livestream_gui

        # store the default render mode
        if not self._has_gui and not self._offscreen_render:
            # set default render mode
            # note: this is the terminal state: cannot exit from this render mode
            self.render_mode = self.RenderMode.NO_GUI_OR_RENDERING
            # set viewport context to None
            self._viewport_context = None
            self._viewport_window = None
        elif not self._has_gui and self._offscreen_render:
            # set default render mode
            # note: this is the terminal state: cannot exit from this render mode
            self.render_mode = self.RenderMode.PARTIAL_RENDERING
            # set viewport context to None
            self._viewport_context = None
            self._viewport_window = None
        else:
            # note: need to import here in case the UI is not available (ex. headless mode)
            import omni.ui as ui
            from omni.kit.viewport.utility import get_active_viewport

            # set default render mode
            # note: this can be changed by calling the `set_render_mode` function
            self.render_mode = self.RenderMode.FULL_RENDERING
            # acquire viewport context
            self._viewport_context = get_active_viewport()
            self._viewport_context.updates_enabled = True  # type: ignore
            # acquire viewport window
            # TODO @mayank: Why not just use get_active_viewport_and_window() directly?
            self._viewport_window = ui.Workspace.get_window("Viewport")
            # counter for periodic rendering
            self._render_throttle_counter = 0
            # rendering frequency in terms of number of render calls
            self._render_throttle_period = 5

        # check the case where we don't need to render the viewport
        # since render_viewport can only be False in headless mode, we only need to check for offscreen_render
        if not self._render_viewport and self._offscreen_render:
            # disable the viewport if offscreen_render is enabled
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().updates_enabled = False

        # override enable scene querying if rendering is enabled
        # this is needed for some GUI features
        if self._has_gui:
            self.cfg.enable_scene_query_support = True

        # create a tensor for gravity
        # note: this line is needed to create a "tensor" in the device to avoid issues with torch 2.1 onwards.
        #   the issue is with some heap memory corruption when torch tensor is created inside the asset class.
        #   you can reproduce the issue by commenting out this line and running the test `test_articulation.py`.
        self._gravity_tensor = torch.tensor(self.cfg.gravity, dtype=torch.float32, device=self.cfg.device)

        # add callback to deal the simulation app when simulation is stopped.
        # this is needed because physics views go invalid once we stop the simulation
        if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
            self._app_control_on_stop_handle = timeline_event_stream.create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP),
                lambda *args, obj=weakref.proxy(self): obj._app_control_on_stop_callback(*args),
                order=15,
            )
        else:
            self._app_control_on_stop_handle = None

        # obtain the simulation interfaces
        self._app = omni.kit.app.get_app_interface()
        self._timeline = omni.timeline.get_timeline_interface()
        self._loop_runner = omni_loop.acquire_loop_interface()
        self._physx_sim_iface = omni.physx.get_physx_simulation_interface()

        # create a dummy physics simulation view
        self._physics_sim_view = None
        # set the simulation to auto-update
        self._timeline.set_auto_update(True)
        # set simulation parameters
        self._backend = "torch"
        # set parameters for physics simulation
        self._init_stage()

    def __new__(cls, *args, **kwargs) -> SimulationContext:
        """Creates a new instance of the simulation context.

        To ensure that only one instance of the simulation context is created, this method overrides the
        default constructor and enforces the singleton pattern.
        """
        if SimulationContext._instance is None:
            SimulationContext._instance = super().__new__(cls)
        else:
            carb.log_info("An instance of the simulation context already exists. Returning the existing instance.")
        return SimulationContext._instance

    """
    Properties.
    """

    @property
    def app(self) -> omni.kit.app.IApp:
        """An instance of the Omniverse Kit Application interface."""
        return self._app

    @property
    def stage(self) -> Usd.Stage:
        """The current open USD stage."""
        return omni.usd.get_context().get_stage()

    @property
    def backend(self) -> Literal["numpy", "torch", "warp"]:
        """The current computational backend used for simulation."""
        return self._backend  # type: ignore

    @property
    def device(self) -> str:
        """The device used for simulation."""
        return self.cfg.device

    """
    Operations - Attributes.
    """

    def has_gui(self) -> bool:
        """Returns whether the simulation has a GUI enabled.

        True if the simulation has a GUI enabled either locally or live-streamed.
        """
        return self._has_gui

    def has_rtx_sensors(self) -> bool:
        """Returns whether the simulation has any RTX-rendering related sensors.

        This function returns the value of the simulation parameter ``"/isaaclab/render/rtx_sensors"``.
        The parameter is set to True when instances of RTX-related sensors (cameras or LiDARs) are
        created using Isaac Lab's sensor classes.

        True if the simulation has RTX sensors (such as USD Cameras or LiDARs).

        For more information, please check `NVIDIA RTX documentation`_.

        .. _NVIDIA RTX documentation: https://developer.nvidia.com/rendering-technologies
        """
        return self._settings_iface.get_as_bool("/isaaclab/render/rtx_sensors")

    def is_fabric_enabled(self) -> bool:
        """Returns whether the fabric interface is enabled.

        When fabric interface is enabled, USD read/write operations are disabled. Instead all applications
        read and write the simulation state directly from the fabric interface. This reduces a lot of overhead
        that occurs during USD read/write operations.

        For more information, please check `Fabric documentation`_.

        .. _Fabric documentation: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html
        """
        return self._physx_fabric_iface is not None

    def get_physics_dt(self) -> float:
        """Returns the physics time-step used for simulation."""
        return self.cfg.dt

    def get_version(self) -> tuple[int, int, int]:
        """Returns the version of the simulator.

        This is a wrapper around the ``omni.isaac.version.get_version()`` function.

        The returned tuple contains the following information:

        * Major version (int): This is the year of the release (e.g. 2022).
        * Minor version (int): This is the half-year of the release (e.g. 1 or 2).
        * Patch version (int): This is the patch number of the release (e.g. 0).

        """
        if not hasattr(self, "_isaacsim_version"):
            import omni.isaac.version

            self._isaacsim_version = omni.isaac.version.get_version()

        return int(self._isaacsim_version[2]), int(self._isaacsim_version[3]), int(self._isaacsim_version[4])

    """
    Operations - New utilities.
    """

    @staticmethod
    def set_camera_view(
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        camera_prim_path: str = "/OmniverseKit_Persp",
    ):
        """Set the location and target of the viewport camera in the stage.

        Note:
            This is a wrapper around the :meth:`omni.isaac.core.utils.viewports.set_camera_view` function.
            It is provided here for convenience to reduce the amount of imports needed.

        Args:
            eye: The location of the camera eye.
            target: The location of the camera target.
            camera_prim_path: The path to the camera primitive in the stage. Defaults to
                "/OmniverseKit_Persp".
        """
        set_camera_view(eye, target, camera_prim_path)

    def set_render_mode(self, mode: RenderMode):
        """Change the current render mode of the simulation.

        Please see :class:`RenderMode` for more information on the different render modes.

        .. note::
            When no GUI is available (locally or livestreamed), we do not need to choose whether the viewport
            needs to render or not (since there is no GUI). Thus, in this case, calling the function will not
            change the render mode.

        Args:
            mode (RenderMode): The rendering mode. If different than SimulationContext's rendering mode,
            SimulationContext's mode is changed to the new mode.

        Raises:
            ValueError: If the input mode is not supported.
        """
        # check if mode change is possible -- not possible when no GUI is available
        if not self._has_gui:
            carb.log_warn(
                f"Cannot change render mode when GUI is disabled. Using the default render mode: {self.render_mode}."
            )
            return
        # check if there is a mode change
        # note: this is mostly needed for GUI when we want to switch between full rendering and no rendering.
        if mode != self.render_mode:
            if mode == self.RenderMode.FULL_RENDERING:
                # display the viewport and enable updates
                self._viewport_context.updates_enabled = True  # type: ignore
                self._viewport_window.visible = True  # pyright: ignore [reportOptionalMemberAccess]
            elif mode == self.RenderMode.PARTIAL_RENDERING:
                # hide the viewport and disable updates
                self._viewport_context.updates_enabled = False  # type: ignore
                self._viewport_window.visible = False  # type: ignore
            elif mode == self.RenderMode.NO_RENDERING:
                # hide the viewport and disable updates
                if self._viewport_context is not None:
                    self._viewport_context.updates_enabled = False  # type: ignore
                    self._viewport_window.visible = False  # pyright: ignore [reportOptionalMemberAccess]
                # reset the throttle counter
                self._render_throttle_counter = 0
            else:
                raise ValueError(f"Unsupported render mode: {mode}! Please check `RenderMode` for details.")
            # update render mode
            self.render_mode = mode

    def set_setting(self, name: str, value: Any):
        """Set simulation settings using the Carbonite SDK.

        .. note::
            If the input setting name does not exist, it will be created. If it does exist, the value will be
            overwritten. Please make sure to use the correct setting name.

            To understand the settings interface, please refer to the
            `Carbonite SDK <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/settings.html>`_
            documentation.

        Args:
            name: The name of the setting.
            value: The value of the setting.
        """
        self._settings_iface.set(name, value)

    def get_setting(self, name: str) -> Any:
        """Read the simulation setting using the Carbonite SDK.

        Args:
            name: The name of the setting.

        Returns:
            The value of the setting.
        """
        return self._settings_iface.get(name)

    """
    Operations - Timeline.
    """

    def is_playing(self) -> bool:
        """Check whether the simulation is playing.

        Returns:
            True if the simulator is playing.
        """
        return self._timeline.is_playing()

    def is_stopped(self) -> bool:
        """Check whether the simulation is stopped.

        Returns:
            True if the simulator is stopped.
        """
        return self._timeline.is_stopped()

    def play(self) -> None:
        """Start playing the simulation."""
        # play the simulation
        self._timeline.play()
        # perform additional render to update the UI
        self.render()

    def pause(self) -> None:
        """Pause the simulation."""
        self._timeline.pause()
        # perform additional render to update the UI
        self.render()

    def stop(self) -> None:
        """Stops the simulation."""
        self._timeline.stop()
        # detach the stage from physics simulation
        self._physx_sim_iface.detach_stage()
        if self._physx_fabric_iface is not None:
            self._physx_fabric_iface.detach_stage()
        # perform additional render to update the UI
        self.render()

    def reset(self, soft: bool = False):
        """Resets and initializes the simulation.

        This method initializes the handles for the physics simulation and the rendering components.

        In a soft reset, the simulation is not stopped and the physics simulation is not reset. This is useful when
        you want to reset the simulation without stopping the simulation.

        Args:
            soft: Whether to perform a soft reset. Defaults to False.
        """
        if not soft:
            # stop the simulation if it is running to reset
            if not self.is_stopped():
                carb.log_warn("Stopping the simulation before resetting.")
                self.stop()
            # attach the USD stage to physics simulation
            # note: we detach first to ensure that the stage is not attached multiple times
            stage_id = UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
            # -- simulation interface
            self._physx_sim_iface.attach_stage(stage_id)
            # -- fabric interface
            if self._physx_fabric_iface is not None:
                self._physx_fabric_iface.attach_stage(stage_id)

            # play the simulation to reset the physics simulation
            self.play()
            # perform one simulation step to initialize the physics simulation
            self.step(render=True)

            # create physics simulation view
            self._physics_sim_view = omni.physics.tensors.create_simulation_view(self.backend)
            self._physics_sim_view.set_subspace_roots("/")

            # perform additional rendering steps to warm up replicator buffers
            # this is only needed for the first time we set the simulation
            for _ in range(2):
                self.render()
        else:
            if not hasattr(self, "_physics_sim_view"):
                raise RuntimeError("Physics simulation view does not exist. Please perform a hard reset.")

    def step(self, render: bool = True):
        """Steps the simulation.

        .. note::
            This function blocks if the timeline is paused. It only returns when the timeline is playing.

        Args:
            render: Whether to render the scene after stepping the physics simulation.
                If set to False, the scene is not rendered and only the physics simulation is stepped.
        """
        # check if the simulation timeline is paused. in that case keep stepping until it is playing
        if not self.is_playing():
            # step the simulator (but not the physics) to have UI still active
            while not self.is_playing():
                self.render()
                # meantime if someone stops, break out of the loop
                if self.is_stopped():
                    break
            # need to do one step to refresh the app
            # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
            #   without this the app becomes unresponsive.
            # FIXME: This steps physics as well, which we is not good in general.
            self.app.update()

        # step the simulation
        if render:
            # physics dt is zero, no need to step physics, just render
            if self.cfg.dt == 0.0:
                self.render()
            else:
                self._app.update()
        else:
            self._physx_sim_iface.simulate(self.cfg.dt, 0.0)
            self._physx_sim_iface.fetch_results()

    def render(self, mode: RenderMode | None = None):
        """Refreshes the rendering components including UI elements and view-ports depending on the render mode.

        This function is used to refresh the rendering components of the simulation. This includes updating the
        view-ports, UI elements, and other extensions (besides physics simulation) that are running in the
        background. The rendering components are refreshed based on the render mode.

        Please see :class:`RenderMode` for more information on the different render modes.

        Args:
            mode: The rendering mode. Defaults to None, in which case the current rendering mode is used.
        """
        # check if we need to change the render mode
        if mode is not None:
            self.set_render_mode(mode)
        # render based on the render mode
        if self.render_mode == self.RenderMode.NO_GUI_OR_RENDERING:
            # we never want to render anything here (this is for complete headless mode)
            pass
        elif self.render_mode == self.RenderMode.NO_RENDERING:
            # throttle the rendering frequency to keep the UI responsive
            self._render_throttle_counter += 1
            if self._render_throttle_counter % self._render_throttle_period == 0:
                self._render_throttle_counter = 0
                # here we don't render viewport so don't need to flush fabric data
                # note: we don't call super().render() anymore because they do flush the fabric data
                self.set_setting("/app/player/playSimulations", False)
                self._app.update()
                self.set_setting("/app/player/playSimulations", True)
        else:
            # manually flush the fabric data to update Hydra textures
            if self._physx_fabric_iface is not None:
                if self._physics_sim_view is not None and self.is_playing():
                    # Update the articulations' link's poses before rendering
                    self._physics_sim_view.update_articulations_kinematic()
                self._update_fabric(0.0, 0.0)
            # render the simulation
            # note: we don't call super().render() anymore because they do above operation inside
            #  and we don't want to do it twice. We may remove it once we drop support for Isaac Sim 2022.2.
            self.set_setting("/app/player/playSimulations", False)
            self._app.update()
            self.set_setting("/app/player/playSimulations", True)

    """
    Operations - Instance handling.
    """

    @classmethod
    def instance(cls) -> SimulationContext | None:
        """Returns the instance of the simulation context.

        The returned instance is the singleton instance of the simulation context. If the instance does not exist,
        it returns None.
        """
        return SimulationContext._instance

    @classmethod
    def clear_instance(cls):
        # clear the callback
        if cls._instance is not None:
            # clear callback for shutting down the app
            if cls._instance._app_control_on_stop_handle is not None:
                cls._instance._app_control_on_stop_handle.unsubscribe()
                cls._instance._app_control_on_stop_handle = None
            # detach the stage from physics simulation
            cls._instance._physx_sim_iface.detach_stage()
            if cls._instance._physx_fabric_iface is not None:
                cls._instance._physx_fabric_iface.detach_stage()
            # set instance to None
            cls._instance = None
            cls._is_initialized = False

    """
    Helper Functions
    """

    def _init_stage(self):
        """Initializes the stage for the simulation."""
        # check if stage exists
        if self.stage is None:
            carb.log_info("Stage does not exist. Creating a new stage.")
            omni.usd.get_context().new_stage()

        # ensure the stage has "z-up" orientation and 1.0 unit scale
        # we follow this convention to be consistent with robotics and other simulators
        with Usd.EditContext(self.stage, self.stage.GetRootLayer()):
            UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        # check that the prim path for physics scene is valid
        if not Sdf.Path(self.cfg.physics_prim_path).IsAbsolutePath():
            raise ValueError(f"Physics prim path {self.cfg.physics_prim_path} is not an absolute path.")
        # find the physics scene in the stage if it exists
        # note: since the stage is pretty empty in the beginning, we can just traverse the stage without optimization
        physics_scene_prim = None
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                carb.log_warn(f"Physics scene already exists in the stage at path '{prim.GetPath()}'. Reusing it.")
                physics_scene_prim = prim
                break
        # create physics scene if it doesn't exist
        if not physics_scene_prim:
            physics_scene_prim = self.stage.DefinePrim(self.cfg.physics_prim_path, "PhysicsScene")
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
        else:
            if physics_scene_prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                physx_scene_api = PhysxSchema.PhysxSceneAPI(physics_scene_prim)
            else:
                physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
        # convert the prim to physics scene
        physics_scene_api = UsdPhysics.Scene(physics_scene_prim)

        # resolve the simulation device
        self.cfg.device = self._resolve_simulation_device(self.cfg.device)
        # configure physx parameters and bind material
        self._configure_physics_params(physics_scene_api, physx_scene_api)

        # a stage update here is needed for the case when physics_dt != rendering_dt, otherwise the app crashes
        # when in headless mode
        self.set_setting("/app/player/playSimulations", False)
        self.app.update()
        self.set_setting("/app/player/playSimulations", True)

        # load flatcache/fabric interface
        self._load_fabric_interface()

    def _resolve_simulation_device(self, device: str) -> str:
        """Resolves the device to use for simulation and returns the resolved device."""
        if "cuda" in self.cfg.device:
            # check if cuda is available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device is not available. Please check your device configuration.")
            # check if the device is available
            if device == "cuda":
                device_id = self._settings_iface.get_as_int("/physics/cudaDevice")
                # in-case device ID is not set, use the first available device
                if device_id < 0:
                    device_id = 0
                    self._settings_iface.set("/physics/cudaDevice", device_id)
                # modify the device string
                device = f"cuda:{device_id}"
            else:
                # set the device to the specified device
                device_id = int(self.cfg.device.split(":")[-1])
                self._settings_iface.set("/physics/cudaDevice", device_id)
        elif device == "cpu":
            # set the device to CPU
            self._settings_iface.set("/physics/cudaDevice", -1)
        else:
            raise ValueError(f"Unsupported device: {device}! Please check the device configuration.")

        # return the resolved device
        return device

    def _configure_physics_params(self, physics_scene: UsdPhysics.Scene, physx_scene_api: PhysxSchema.PhysxSceneAPI):
        """Sets the physics parameters for the simulation."""
        # change dispatcher to use the default dispatcher in PhysX SDK instead of carb tasking
        # note: dispatcher handles how threads are launched for multi-threaded physics
        self._settings_iface.set_bool("/physics/physxDispatcher", True)
        # disable contact processing in omni.physx if requested
        # note: helpful when creating contact reporting over limited number of objects in the scene
        self._settings_iface.set_bool("/physics/disableContactProcessing", self.cfg.disable_contact_processing)
        # enable custom geometry for cylinder and cone collision shapes to allow contact reporting for them
        # reason: cylinders and cones aren't natively supported by PhysX so we need to use custom geometry flags
        # reference: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html?highlight=capsule#geometry
        self._settings_iface.set_bool("/physics/collisionConeCustomGeometry", False)
        self._settings_iface.set_bool("/physics/collisionCylinderCustomGeometry", False)

        # Gravity
        # note: Isaac sim only takes the "up-axis" as the gravity direction. But physics allows any direction so we
        #  need to convert the gravity vector to a direction and magnitude pair explicitly.
        gravity = np.asarray(self.cfg.gravity)
        gravity_magnitude = np.linalg.norm(gravity)

        # Avoid division by zero
        if gravity_magnitude != 0.0:
            gravity_direction = gravity / gravity_magnitude
        else:
            gravity_direction = gravity

        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_direction))
        physics_scene.CreateGravityMagnitudeAttr(gravity_magnitude)

        # create the default physics material
        # this material is used when no material is specified for a primitive
        # check: https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/simulation-control/physics-settings.html#physics-materials
        material_path = f"{self.cfg.physics_prim_path}/defaultMaterial"
        self.cfg.physics_material.func(material_path, self.cfg.physics_material)
        # bind the physics material to the scene
        bind_physics_material(self.cfg.physics_prim_path, material_path)

        # CPU/GPU device based settings
        # -- Collision detection
        broadphase_type = "GPU" if "cuda" in self.cfg.device else "MBP"
        physx_scene_api.CreateBroadphaseTypeAttr(broadphase_type)
        # -- Dynamics
        enable_gpu_dynamics = "cuda" in self.cfg.device
        physx_scene_api.CreateEnableGPUDynamicsAttr(enable_gpu_dynamics)
        # -- Physics pipeline
        use_gpu_pipeline = "cuda" in self.cfg.device
        self._settings_iface.set_bool("/physics/suppressReadback", use_gpu_pipeline)

        # PhysX specific settings
        # -- Scene query support
        physx_scene_api.CreateEnableSceneQuerySupportAttr(self.cfg.enable_scene_query_support)
        # -- Solver type
        physx_scene_api.CreateSolverTypeAttr(self.cfg.physx.solver_type)
        # -- Position iteration count
        physx_scene_api.CreateMinPositionIterationCountAttr(self.cfg.physx.min_position_iteration_count)
        physx_scene_api.CreateMaxPositionIterationCountAttr(self.cfg.physx.max_position_iteration_count)
        # -- Velocity iteration count
        physx_scene_api.CreateMinVelocityIterationCountAttr(self.cfg.physx.min_velocity_iteration_count)
        physx_scene_api.CreateMaxVelocityIterationCountAttr(self.cfg.physx.max_velocity_iteration_count)
        # -- Continuous Collision Detection (CCD)
        # ref: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html?highlight=ccd#continuous-collision-detection
        physx_scene_api.CreateEnableCCDAttr(self.cfg.physx.enable_ccd)
        # -- Stabilization pass
        physx_scene_api.CreateEnableStabilizationAttr(self.cfg.physx.enable_stabilization)
        # -- Enhanced determinism
        physx_scene_api.CreateEnableEnhancedDeterminismAttr(self.cfg.physx.enable_enhanced_determinism)
        # -- Contact bounce velocity threshold
        physx_scene_api.CreateBounceThresholdAttr(self.cfg.physx.bounce_threshold_velocity)
        # -- Friction offset threshold
        physx_scene_api.CreateFrictionOffsetThresholdAttr(self.cfg.physx.friction_offset_threshold)
        # -- Friction correlation distance
        physx_scene_api.CreateFrictionCorrelationDistanceAttr(self.cfg.physx.friction_correlation_distance)

        # PhysX buffer sizes for GPU preallocation
        if "cuda" in self.cfg.device:
            # -- max rigid contact count
            physx_scene_api.CreateGpuMaxRigidContactCountAttr(self.cfg.physx.gpu_max_rigid_contact_count)
            # -- max rigid patch count
            physx_scene_api.CreateGpuMaxRigidPatchCountAttr(self.cfg.physx.gpu_max_rigid_patch_count)
            # -- found lost pairs capacity
            physx_scene_api.CreateGpuFoundLostPairsCapacityAttr(self.cfg.physx.gpu_found_lost_pairs_capacity)
            # -- found lost aggregate pairs capacity
            physx_scene_api.CreateGpuFoundLostAggregatePairsCapacityAttr(
                self.cfg.physx.gpu_found_lost_aggregate_pairs_capacity
            )
            # -- total aggregate pairs capacity
            physx_scene_api.CreateGpuTotalAggregatePairsCapacityAttr(self.cfg.physx.gpu_total_aggregate_pairs_capacity)
            # -- collision stack size
            physx_scene_api.CreateGpuCollisionStackSizeAttr(self.cfg.physx.gpu_collision_stack_size)
            # -- heap capacity
            physx_scene_api.CreateGpuHeapCapacityAttr(self.cfg.physx.gpu_heap_capacity)
            # -- temp buffer capacity
            physx_scene_api.CreateGpuTempBufferCapacityAttr(self.cfg.physx.gpu_temp_buffer_capacity)
            # -- max num partitions
            physx_scene_api.CreateGpuMaxNumPartitionsAttr(self.cfg.physx.gpu_max_num_partitions)
            # -- max soft body contacts
            physx_scene_api.CreateGpuMaxSoftBodyContactsAttr(self.cfg.physx.gpu_max_soft_body_contacts)
            # -- max particle contacts
            physx_scene_api.CreateGpuMaxParticleContactsAttr(self.cfg.physx.gpu_max_particle_contacts)

        # Physics time-step
        if self.cfg.render_interval <= 0:
            carb.log_warn("Render interval is zero or negative. Setting it to 1.")
            self.cfg.render_interval = 1
        if self.cfg.dt < 0:
            raise ValueError(f"Physics time-step cannot be negative. Received: {self.cfg.dt}")
        if self.cfg.dt == 0:
            carb.log_warn("Physics time-step is zero. Physics simulation will be disabled.")
            steps_per_second = 0
            min_frame_rate = 0
        else:
            steps_per_second = int(1 / self.cfg.dt)
            min_frame_rate = int(steps_per_second / self.cfg.render_interval)
        # set the time-step parameters
        physx_scene_api.CreateTimeStepsPerSecondAttr(steps_per_second)
        self._settings_iface.set_int("/persistent/simulation/minFrameRate", min_frame_rate)

        # Rendering time-step
        rendering_dt = self.cfg.render_interval * self.cfg.dt
        rendering_hz = int(1 / rendering_dt)
        # -- stage settings
        with Usd.EditContext(self.stage, self.stage.GetRootLayer()):
            self.stage.SetTimeCodesPerSecond(rendering_hz)
        # -- timeline settings
        self._timeline.set_target_framerate(rendering_hz)
        # -- isaac sim's loop runner settings
        if self._loop_runner is not None:
            # THINK: we don't need to rate-limit the app when running headless
            self._settings_iface.set_bool("/app/runLoops/main/rateLimitEnabled", builtins.ISAAC_LAUNCHED_FROM_TERMINAL)
            self._settings_iface.set_int("/app/runLoops/main/rateLimitFrequency", rendering_hz)
            # set the manual step size to the rendering time-step
            self._loop_runner.set_manual_step_size(rendering_dt)
            self._loop_runner.set_manual_mode(True)

    def _load_fabric_interface(self):
        """Loads the fabric interface if enabled."""
        # enable the extension for fabric interface
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.physx.fabric", self.cfg.use_fabric)

        # disable USD updates if fabric is enabled
        self._settings_iface.set_bool("/physics/updateToUsd", not self.cfg.use_fabric)
        self._settings_iface.set_bool("/physics/updateParticlesToUsd", not self.cfg.use_fabric)
        self._settings_iface.set_bool("/physics/updateVelocitiesToUsd", not self.cfg.use_fabric)
        self._settings_iface.set_bool("/physics/updateForceSensorsToUsd", not self.cfg.use_fabric)
        self._settings_iface.set_bool("/physics/outputVelocitiesLocalSpace", not self.cfg.use_fabric)

        # load the fabric interface if enabled
        if self.cfg.use_fabric:
            from omni.physxfabric import get_physx_fabric_interface

            # acquire fabric interface
            self._physx_fabric_iface = get_physx_fabric_interface()
            if hasattr(self._physx_fabric_iface, "force_update"):
                # The update method in the fabric interface only performs an update if a physics step has occurred.
                # However, for rendering, we need to force an update since any element of the scene might have been
                # modified in a reset (which occurs after the physics step) and we want the renderer to be aware of
                # these changes.
                self._update_fabric = self._physx_fabric_iface.force_update
            else:
                # Needed for backward compatibility with older Isaac Sim versions
                self._update_fabric = self._physx_fabric_iface.update
        else:
            # set the fabric interface to None
            self._physx_fabric_iface = None
            self._update_fabric = lambda *args, **kwargs: None

    """
    Callbacks.
    """

    def _app_control_on_stop_callback(self, event: carb.events.IEvent):
        """Callback to deal with the app when the simulation is stopped.

        Once the simulation is stopped, the physics handles go invalid. After that, it is not possible to
        resume the simulation from the last state. This leaves the app in an inconsistent state, where
        two possible actions can be taken:

        1. **Keep the app rendering**: In this case, the simulation is kept running and the app is not shutdown.
           However, the physics is not updated and the script cannot be resumed from the last state. The
           user has to manually close the app to stop the simulation.
        2. **Shutdown the app**: This is the default behavior. In this case, the app is shutdown and
           the simulation is stopped.

        Note:
            This callback is used only when running the simulation in a standalone python script. In an extension,
            it is expected that the user handles the extension shutdown.
        """
        # check if the simulation is stopped
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # keep running the simulator when configured to not shutdown the app
            if self._has_gui and sys.exc_info()[0] is None:
                self.app.print_and_log(
                    "Simulation is stopped. The app will keep running with physics disabled."
                    " Press Ctrl+C or close the window to exit the app."
                )
                while self.app.is_running():
                    self.render()

        # Note: For the following code:
        #   The method is an exact copy of the implementation in the `omni.isaac.kit.SimulationApp` class.
        #   We need to remove this method once the SimulationApp class becomes a singleton.

        # make sure that any replicator workflows finish rendering/writing
        try:
            import omni.replicator.core as rep

            rep_status = rep.orchestrator.get_status()
            if rep_status not in [rep.orchestrator.Status.STOPPED, rep.orchestrator.Status.STOPPING]:
                rep.orchestrator.stop()
            if rep_status != rep.orchestrator.Status.STOPPED:
                rep.orchestrator.wait_until_complete()

            # Disable capture on play to avoid replicator engaging on any new timeline events
            rep.orchestrator.set_capture_on_play(False)
        except Exception:
            pass

        # workaround for exit issues, clean the stage first:
        if omni.usd.get_context().can_close_stage():
            omni.usd.get_context().close_stage()

        # print logging information
        self.app.print_and_log("Simulation is stopped. Shutting down the app...")

        # Cleanup any running tracy instances so data is not lost
        try:
            profiler_tracy = carb.profiler.acquire_profiler_interface(plugin_name="carb.profiler-tracy.plugin")
            if profiler_tracy:
                profiler_tracy.set_capture_mask(0)
                profiler_tracy.end(0)
                profiler_tracy.shutdown()
        except RuntimeError:
            # Tracy plugin was not loaded, so profiler never started - skip checks.
            pass

        # Disable logging before shutdown to keep the log clean
        # Warnings at this point don't matter as the python process is about to be terminated
        logging = carb.logging.acquire_logging()
        logging.set_level_threshold(carb.logging.LEVEL_ERROR)

        # App shutdown is disabled to prevent crashes on shutdown. Terminating carb is faster
        # self._app.shutdown()
        self._framework.unload_all_plugins()


##
# Context Manager for Simulation.
##


@contextmanager
def build_simulation_context(
    create_new_stage: bool = True,
    gravity_enabled: bool = True,
    device: str = "cuda:0",
    dt: float = 0.01,
    sim_cfg: SimulationCfg | None = None,
    add_ground_plane: bool = False,
    add_lighting: bool = False,
    auto_add_lighting: bool = False,
) -> Iterator[SimulationContext]:
    """Context manager to build a simulation context with the provided settings.

    This function facilitates the creation of a simulation context and provides flexibility in configuring various
    aspects of the simulation, such as time step, gravity, device, and scene elements like ground plane and
    lighting.

    If :attr:`sim_cfg` is None, then an instance of :class:`SimulationCfg` is created with default settings, with parameters
    overwritten based on arguments to the function.

    An example usage of the context manager function:

    ..  code-block:: python

        with build_simulation_context() as sim:
             # Design the scene

             # Play the simulation
             sim.reset()
             while sim.is_playing():
                 sim.step()

    Args:
        create_new_stage: Whether to create a new stage. Defaults to True.
        gravity_enabled: Whether to enable gravity in the simulation. Defaults to True.
        device: Device to run the simulation on. Defaults to "cuda:0".
        dt: Time step for the simulation: Defaults to 0.01.
        sim_cfg: :class:`omni.isaac.lab.sim.SimulationCfg` to use for the simulation. Defaults to None.
        add_ground_plane: Whether to add a ground plane to the simulation. Defaults to False.
        add_lighting: Whether to add a dome light to the simulation. Defaults to False.
        auto_add_lighting: Whether to automatically add a dome light to the simulation if the simulation has a GUI.
            Defaults to False. This is useful for debugging tests in the GUI.

    Yields:
        The simulation context to use for the simulation.

    """
    try:
        if create_new_stage:
            omni.usd.get_context().new_stage()

        if sim_cfg is None:
            # Construct one and overwrite the dt, gravity, and device
            sim_cfg = SimulationCfg(dt=dt)

            # Set up gravity
            if gravity_enabled:
                sim_cfg.gravity = (0.0, 0.0, -9.81)
            else:
                sim_cfg.gravity = (0.0, 0.0, 0.0)

            # Set device
            sim_cfg.device = device

        # Construct simulation context
        sim = SimulationContext(sim_cfg)

        if add_ground_plane:
            # Ground-plane
            cfg = GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg)

        if add_lighting or (auto_add_lighting and sim.has_gui()):
            # Lighting
            cfg = DomeLightCfg(
                color=(0.1, 0.1, 0.1),
                enable_color_temperature=True,
                color_temperature=5500,
                intensity=10000,
            )
            # Dome light named specifically to avoid conflicts
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        yield sim

    except Exception:
        carb.log_error(traceback.format_exc())
        raise
    finally:
        if not sim.has_gui():
            # Stop simulation only if we aren't rendering otherwise the app will hang indefinitely
            sim.stop()

        # Clear sim instance
        sim.clear_instance()
