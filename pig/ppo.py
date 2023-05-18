import os
import torch

"""
    Continuous PPO implementation.

    The network structure is not configurable now
"""

"""Neural network in PPO. PPO is actor-critic architecture, 
    so there are two nets:
        - Actor / Policy Net. In continuous scenario, it returns a distribution.
        - Critic / Value Net.
    """

class Actor(torch.nn.Module):
    def __init__(self, num_observations, hidden_size, num_actions):
        super(Actor, self).__init__()
        self.policy_shared = torch.nn.Sequential(
            torch.nn.Linear(num_observations, hidden_size),
            torch.nn.LeakyReLU()
        )
        self.policy_mu = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_actions),
            torch.nn.Tanh()
        )
        self.policy_sigma = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_actions),
            torch.nn.Softplus()
        )
    
    def forward(self, input):
        shared = self.policy_shared(input)
        mu = 2 * self.policy_mu(shared)
        sigma = self.policy_sigma(shared)
        return mu, sigma


class Critic(torch.nn.Module):
    def __init__(self, num_observations, hidden_size, num_actions):
        super(Critic, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_observations, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, num_actions),
        )
    
    def forward(self, input):
        return self.net(input)


class PPO:
    """PPO Agent."""

    def __init__(self, cfg, env, save_checkpoint_path):
        self.load_checkpoint_path = cfg.checkpoint
        print(self.load_checkpoint_path)
        self.save_checkpoint_path = save_checkpoint_path

        self.learning_rate = cfg.train.params.config.learning_rate
        self.gamma = cfg.train.params.config.gamma
        self.lmbda = cfg.train.params.config.tau
        self.minibatch_size = cfg.train.params.config.minibatch_size
        self.e_clip = cfg.train.params.config.e_clip

        self.max_epochs = cfg.train.params.config.max_epochs
        self.horizon_length = cfg.train.params.config.horizon_length
        self.mini_epochs = cfg.train.params.config.mini_epochs
        self.save_frequency = cfg.train.params.config.save_frequency 

        self.env = env
        self.num_envs = cfg.task.env.numEnvs

        self.rl_device = cfg.rl_device

        self.actor = Actor(env.num_observations, 128, env.num_actions).to(self.rl_device)
        self.critic = Critic(env.num_observations, 128, env.num_actions).to(self.rl_device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-4)#self.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=5e-3)#self.learning_rate)


    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float32).to(self.rl_device)


    def take_action(self, obs):
        mu, sigma = self.actor(obs)
        dist = torch.distributions.Normal(mu, sigma)
        return dist.sample()


    def update(self, obs_batch, act_batch, rew_batch, next_obs_batch, dones_batch):
        rew_batch = rew_batch.unsqueeze(-1)
        dones_batch = dones_batch.unsqueeze(-1)

        td_target = rew_batch + self.gamma * (1 - dones_batch) * self.critic(next_obs_batch)
        del rew_batch, dones_batch, next_obs_batch
        td_delta = td_target - self.critic(obs_batch)
        advantage = self.compute_advantage(td_delta)
        mu, sigma = self.actor(obs_batch)
        dist = torch.distributions.Normal(mu.detach(), sigma.detach())
        old_log_probs = dist.log_prob(act_batch)

        for _ in range(self.mini_epochs):
            mu, sigma = self.actor(obs_batch)
            dist = torch.distributions.Normal(mu, sigma)
            log_probs = dist.log_prob(act_batch)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.e_clip, 1+self.e_clip) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(torch.nn.functional.mse_loss(
                self.critic(obs_batch),
                td_target.detach()
            ))

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
    

    def load_model(self, checkpoint_path):
        self.actor = torch.load(checkpoint_path + "/actor.pt", map_location=self.rl_device)
        self.critic = torch.load(checkpoint_path + "/critic.pt", map_location=self.rl_device)


    def save_model(self):
        print(f"Saving model to {self.save_checkpoint_path} ...")
        torch.save(self.actor, self.save_checkpoint_path + f"/actor.pt")
        torch.save(self.critic, self.save_checkpoint_path + f"/critic.pt")
    

    def train(self):
        if not os.path.exists(self.save_checkpoint_path):
            os.makedirs(self.save_checkpoint_path)

        for i in range(self.max_epochs):
            obs_buffer = torch.zeros((self.horizon_length, self.num_envs, self.env.num_observations), dtype=torch.float32).to(self.rl_device)
            actions_buffer = torch.zeros((self.horizon_length, self.num_envs, self.env.num_actions), dtype=torch.float32).to(self.rl_device)
            rewards_buffer = torch.zeros((self.horizon_length, self.num_envs), dtype=torch.float32).to(self.rl_device)
            next_obs_buffer= torch.zeros((self.horizon_length, self.num_envs, self.env.num_observations), dtype=torch.float32).to(self.rl_device)
            dones_buffer = torch.zeros((self.horizon_length, self.num_envs), dtype=torch.float32).to(self.rl_device)
            
            obs = self.env.reset()["obs"]
            rewards_sum = torch.zeros((self.num_envs,), dtype=torch.float32).to(self.rl_device)

            for j in range(self.horizon_length):
                actions = self.take_action(obs)
                next_obs, rewards, dones, _ = self.env.step(actions)
                next_obs = next_obs['obs']

                # dones_idx = dones.nonzero(as_tuple=True)[0]
                # if len(dones_idx) > 0:
                #     self.env.reset_idx(dones_idx)
                #     next_obs = self.env.compute_observations(dones_idx)

                if torch.any(dones):
                    next_obs = self.env.reset()['obs']

                obs_buffer[j] = obs
                actions_buffer[j] = actions
                rewards_buffer[j] = rewards 
                next_obs_buffer[j] = next_obs
                dones_buffer[j] = dones
                obs = next_obs

                rewards_sum += rewards

            indices = torch.randint(self.horizon_length, (self.minibatch_size,), requires_grad=False)
            self.update(obs_buffer[indices], actions_buffer[indices], rewards_buffer[indices], next_obs_buffer[indices], dones_buffer[indices])
            print(f"Episode: {i}, Avg Reward: {torch.mean(rewards_sum / self.horizon_length)}")

            if (i+1) % self.save_frequency == 0:
                self.save_model()


    def eval(self, eval_step):
        if self.load_checkpoint_path:
            self.load_model(self.load_checkpoint_path)

        obs = self.env.reset()["obs"]

        for i in range(eval_step):
            actions = self.take_action(obs)
            next_obs, rewards, dones, _ = self.env.step(actions)
            next_obs = next_obs['obs']
            obs = next_obs
            print(f"Reward: ", torch.mean(rewards))
