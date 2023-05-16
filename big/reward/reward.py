import torch


@torch.jit.script
def height_only_reward(tray_positions, ball_positions, ball_velocities, ball_radius, reset_buf, progress_buf, max_episode_length, 
                    alpha, beta, target_height, height_limit):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, float, float) -> Tuple[Tensor, Tensor]
    
    x, y, z = ball_velocities[..., 0], ball_velocities[..., 1], ball_velocities[..., 2]

    ball_dist = torch.sqrt(alpha * (z - target_height) * (z - target_height) + beta * (x * x + y * y))
    ball_speed = torch.sqrt(ball_velocities[..., 0] * ball_velocities[..., 0] +
                            ball_velocities[..., 1] * ball_velocities[..., 1] +
                            ball_velocities[..., 2] * ball_velocities[..., 2])

    pos_reward = 1.0 / (1.0 + ball_dist)
    reward = pos_reward

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(ball_positions[..., 2] < ball_radius * 1.5, torch.ones_like(reset_buf), reset_buf)

    return reward, reset


@torch.jit.script
def balancing_reward(tray_positions, ball_positions, ball_velocities, ball_radius, reset_buf, progress_buf, max_episode_length, 
                    alpha, beta, target_height, height_limit):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, float, float) -> Tuple[Tensor, Tensor]
    
    x, y, z = ball_velocities[..., 0], ball_velocities[..., 1], ball_velocities[..., 2]

    ball_dist = torch.sqrt(alpha * (z - target_height) * (z - target_height) + beta * (x * x + y * y))
    ball_speed = torch.sqrt(ball_velocities[..., 0] * ball_velocities[..., 0] +
                            ball_velocities[..., 1] * ball_velocities[..., 1] +
                            ball_velocities[..., 2] * ball_velocities[..., 2])

    pos_reward = 1.0 / (1.0 + ball_dist)
    speed_reward = 1.0 / (1.0 + ball_speed)
    reward = pos_reward * speed_reward

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(ball_positions[..., 2] < ball_radius * 1.5, torch.ones_like(reset_buf), reset_buf)

    return reward, reset


@torch.jit.script
def bouncing_reward(tray_positions, ball_positions, ball_velocities, ball_radius, reset_buf, progress_buf, max_episode_length, 
                    alpha, beta, target_height, height_limit):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, float, float) -> Tuple[Tensor, Tensor]
    
    x, y, z = ball_positions[..., 0], ball_positions[..., 1], ball_positions[..., 2]

    ball_dist = torch.sqrt(alpha * (z - target_height) * (z - target_height) + beta * (x * x + y * y))
    ball_speed = torch.sqrt(ball_velocities[..., 0] * ball_velocities[..., 0] +
                            ball_velocities[..., 1] * ball_velocities[..., 1] +
                            ball_velocities[..., 2] * ball_velocities[..., 2])

    pos_reward = 1.0 / (1.0 + ball_dist)
    speed_reward = (1.0 + ball_speed)

    punish_magic = -8 # magic punish number
    punish = torch.where(z > height_limit, punish_magic, 0)

    reward = pos_reward * speed_reward + punish

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(ball_positions[..., 2] < ball_radius * 1.5, torch.ones_like(reset_buf), reset_buf)

    return reward, reset


reward_func_map = {
    "height_only": height_only_reward,
    "balancing":  balancing_reward,
    "bouncing": bouncing_reward
}

def get_reward_by_name(name: str):
    if name not in reward_func_map:
        raise ValueError(f"Unsupported reward func name: {name}")

    return reward_func_map[name]