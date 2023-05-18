from pig.pig import Pig

# import BallBounce task
from big.registry import register_task_map
register_task_map()

"""
    Modify the config path freely!
"""

pig = Pig("big/env/cfg/")
pig.run()
