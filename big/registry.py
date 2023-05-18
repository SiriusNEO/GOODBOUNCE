"""
    Register ball bounce in Isaac Gym envs.
"""

from .env import BallBounce
from isaacgymenvs.tasks import isaacgym_task_map


def register_task_map():
    isaacgym_task_map["BallBounce"] = BallBounce

