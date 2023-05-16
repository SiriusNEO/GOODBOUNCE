"""
    Register ball bounce in Isaac Gym envs.
"""

from .env import BallBounce
from isaacgymenvs.tasks import isaacgym_task_map


def register():
    # overwriting the map since we don't support all envs in IsaacGymEnvs
    isaacgym_task_map["BallBounce"] = BallBounce


def make(*args, **kwargs):
    return isaacgym_task_map["BallBounce"](*args, **kwargs)
