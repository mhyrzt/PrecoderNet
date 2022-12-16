# PrecoderNet

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

PrecoderNet: Hybrid Beamforming for Millimeter Wave Systems with Deep Reinforcement Learning

## Results

## Installation

```bash
git clone git@github.com:mhyrzt/PrecoderNet.git
cd PrecoderNet
pip install -e .
```

## Example

```python

import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from PrecoderNet.ddpg import DDPG
from PrecoderNet.models import Actor, Critic
from PrecoderNet.environemt import Environment, plot_loss
from PrecoderNet.random_process import OrnsteinUhlenbeckProcess

n_s = 6
n_r = 32

channel_matrix = np.random.randn(n_r, n_s) # Change This However You like :)) Based on Base Station Channel Matrix

CONFIG = {
    "P": 120,
    "var": 1,
    "beta": 0.1,
    "n_t": 128,
    "n_r": n_r,
    "n_s": n_s,
    "n_t_rf": 6,
    "n_r_rf": 6,
    "n_cl": 8,
    "n_ray": 10,
    "v_rf_a": 100,
    "v_rf_iteration": 1000,
    "channel_matrix": channel_matrix
}

env = Environment(**CONFIG)
plot_loss(env.v_rf, env.v_bb, env.v_rf_loss)
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
```

## Article

```python
@ARTICLE{9112250,
  author={Wang, Qisheng and Feng, Keming and Li, Xiao and Jin, Shi},
  journal={IEEE Wireless Communications Letters}, 
  title={PrecoderNet: Hybrid Beamforming for Millimeter Wave Systems With Deep Reinforcement Learning}, 
  year={2020},
  volume={9},
  number={10},
  pages={1677-1681},
  doi={10.1109/LWC.2020.3001121}}
```
