from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from PrecoderNet.ddpg import DDPG
from PrecoderNet.environemt import Environment


class Trainer:
    def __init__(self, env: Environment, ddpg: DDPG, epochs) -> None:
        self.env = env
        self.ddpg = ddpg
        self._copy()
        self.epochs = epochs
        self.rewards = []
        self.efficiency = []

    def train(self):
        pbar = tqdm(range(self.epochs), ncols=128)
        state = self.env.get_state()
        for _ in pbar:
            self._copy()
            action = self.ddpg.get_action(state)
            next_state, reward = self.env.step(action)
            if self.env.is_done():
                break

            self \
                .update_ddpg(state, action, reward, next_state) \
                .add_history(reward)

            pbar.set_description(self.log())
            state = next_state
        self.check_constraint()
        return self

    def update_ddpg(self, state, action, reward, next_state):
        self.ddpg.add(state, action, reward, next_state)
        self.ddpg.step()
        return self

    def _copy(self):
        self.env_cpy = deepcopy(self.env)
        self.ddpg_cpy = deepcopy(self.ddpg)

    def add_history(self, reward):
        self.rewards.append(reward)
        self.efficiency.append(self.env.spectral_efficiency())
        return self

    def log(self):
        reward = self.rewards[-1]
        efficiency = self.efficiency[-1]
        constraint = self.env.constraint()
        return f"reward = {round(reward, 3)} | efficiency = {round(efficiency, 3)} | constraint = {round(constraint, 3)}"

    def check_constraint(self):
        if self.env.constraint() <= self.env.P:
            return self
        self.env = self.env_cpy
        self.ddpg = self.ddpg_cpy
        self.rewards.pop()
        self.efficiency.pop()
        return self

    def save_progress_plot(self, file_name: str):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.rewards, "-o", label="Rewards (Upper Bound)")
        ax.plot(self.efficiency, "-o", label="Spectral Efficiency (db)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (Upper Bound)")
        ax.grid()
        ax.legend()
        fig.savefig(file_name)
        return self
