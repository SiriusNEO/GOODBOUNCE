import os
import datetime
import isaacgym
from isaacgymenvs.utils.reformat import omegaconf_to_dict

from .config_parser import parse
from .ppo import PPO


class Pig:
    """Wrapper for IsaacGymEnvs examples. Including:
        - Config parser
        - Environment initialization
        - Testing & Training
    """

    def __init__(self, config_path: str):
        """The config path should be a path to a directory which has a 
        structure like:
            - task/
                - {TaskName}.yaml
            - train/
                - {TaskName}PPO.yaml
            - config.yaml

        The config format refers to IsaacGymEnvs config files:
            https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/
        """

        cfg = parse(config_path)

        from isaacgymenvs.tasks import isaacgym_task_map
        env_creator = isaacgym_task_map[cfg.task_name]

        self.env = env_creator(
            cfg = omegaconf_to_dict(cfg.task),
            sim_device = cfg.sim_device,
            graphics_device_id = cfg.graphics_device_id,
            headless = cfg.headless,
            rl_device = cfg.rl_device,
            virtual_screen_capture = cfg.capture_video,
            force_render = cfg.force_render,
        )

        self.work_dir = "work/"

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.checkpoint_path = f"{self.work_dir}/{cfg.task_name}-{time_str}"

        self.agent = PPO(cfg, self.env, self.checkpoint_path)
        self.test_flag = cfg.test
    

    def run(self):
        if not self.test_flag:
            reward_list = self.agent.train()
            with open(self.checkpoint_path + "/reward.txt", "w") as fp:
                print(reward_list, file=fp)
        else:
            self.agent.eval(100000)