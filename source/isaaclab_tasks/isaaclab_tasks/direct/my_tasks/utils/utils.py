
def get_algorithm(task: str) -> str:
    if "AMP" in task:
        return "AMP"
    elif "PPO" in task:
        return "PPO"
    elif "MOE" in task:
        return "MOE"
    else:
        return ""