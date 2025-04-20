import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import argparse

def animate3D(tensor, highlight_joint: int=0, q: torch.Tensor | None=None, w_last: bool=True):
    if type(tensor) is torch.Tensor: tensor = tensor.numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    minimum = np.min(tensor.reshape(-1, 3), axis=0)
    maximum = np.max(tensor.reshape(-1, 3), axis=0)
    center = 0.5 * (maximum + minimum)
    diff = 0.75 * (maximum - minimum)
            
    ax.set_xlim((center[0] - diff[0], center[0] + diff[0]))
    ax.set_ylim((center[1] - diff[1], center[1] + diff[1]))
    ax.set_zlim((center[2] - diff[2], center[2] + diff[2]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = ax.set_title('Frame: 0')

    scatter = ax.scatter([], [], [], c='red', s=50)
    highlight_scatter = ax.scatter([], [], [], c='blue', s=80) 
    quiver_artist = None  # 初始化前向向量箭头

    def update(frame):
        nonlocal quiver_artist
        
        x = tensor[frame, :, 0]
        y = tensor[frame, :, 1]
        z = tensor[frame, :, 2]
        
        scatter._offsets3d = (np.delete(x, highlight_joint), 
                             np.delete(y, highlight_joint), 
                             np.delete(z, highlight_joint))
        
        highlight_scatter._offsets3d = (x[highlight_joint:highlight_joint+1], 
                                      y[highlight_joint:highlight_joint+1], 
                                      z[highlight_joint:highlight_joint+1])
        
        # 计算并更新前向向量
        if q is not None and frame < q.shape[0]:
            current_q = q[frame]
            if w_last:
                x_q, y_q, z_q, w_q = current_q[..., 0], current_q[..., 1], current_q[..., 2], current_q[..., 3]
            else:
                w_q, x_q, y_q, z_q = current_q[..., 0], current_q[..., 1], current_q[..., 2], current_q[..., 3]
            
            # 计算前向向量
            forward_x = 2 * (x_q*z_q - w_q*y_q)
            forward_y = 2 * (y_q*z_q + w_q*x_q)
            forward_z = 1 - 2 * (x_q*x_q + y_q*y_q)
            
            # 箭头的起点是高亮关节的位置
            start_x = x[highlight_joint]
            start_y = y[highlight_joint]
            start_z = z[highlight_joint]
            
            # 箭头的长度（根据场景动态调整）
            arrow_length = 0.3 * np.linalg.norm(diff)
            
            # 移除旧箭头（如果存在）
            if quiver_artist is not None:
                quiver_artist.remove()
            
            # 添加新箭头
            quiver_artist = ax.quiver(
                start_x, start_y, start_z,
                forward_x * arrow_length, 
                forward_y * arrow_length, 
                forward_z * arrow_length,
                color='blue', arrow_length_ratio=0.2, linewidth=2
            )
        
        title.set_text(f'Frame: {frame}/{tensor.shape[0]}')
        artists = [scatter, highlight_scatter, title]
        if quiver_artist is not None:
            artists.append(quiver_artist)
        return artists

    ani = FuncAnimation(
        fig, update, frames=tensor.shape[0],
        interval=50, blit=False
    )

    plt.show()  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../InterHuman/1.npz")
    args = parser.parse_args()
    data = np.load(args.file)
    
    animate3D(data['body_positions'], highlight_joint=0, q=data['body_rotations'][:,0], w_last=False)