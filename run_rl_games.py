import hydra
from big.registry import register_task_map

from isaacgymenvs.train import launch_rlg_hydra


@hydra.main(config_name="config", config_path="./big/env/cfg")
def launch_rlg_hydra_override(cfg):
    launch_rlg_hydra(cfg)


if __name__ == "__main__":
    register_task_map()
    launch_rlg_hydra_override()
