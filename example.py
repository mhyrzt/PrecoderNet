from config import CONFIG
from PrecoderNet.environemt import Environment, plot_loss
from PrecoderNet.ddpg import DDPG
from PrecoderNet.models import Actor, Critic
from PrecoderNet.random_process import OrnsteinUhlenbeckProcess
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

env = Environment(**CONFIG)
k = env.get_layer_size()

actor = Actor(k, k, (512, 512, 512))
critic = Critic(k, k, (512, 512, 512))
random_process = OrnsteinUhlenbeckProcess(
    size=k,
    theta=0.15,
    mu=0.0,
    sigma=0.2
)
ddpg = DDPG(actor, critic, 10_000, 512, random_process)

count = 600
rewards = []

s = env.get_state()
for _ in (pbar := tqdm(range(count), ncols=128)):
    a = ddpg.get_action(s)
    ns, r = env.step(a)
    if np.isnan(r):
        print("DONE")
        break
    ddpg.add(s, a, r, ns)
    ddpg.step()
    s = ns
    rewards.append(r)
    pbar.set_description(f"reward = {r}")

fig, ax = plt.subplots(dpi=100)
ax.plot(rewards)
ax.set_title("Reward Plot")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward (Upper Bound)")
ax.grid()
plt.show()