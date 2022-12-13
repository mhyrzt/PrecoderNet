from config import CONFIG
from PrecoderNet.environemt import Environment, plot_loss
import numpy as np
env = Environment(**CONFIG)
plot_loss(env.v_rf, env.v_bb, env.v_rf_loss).savefig("v_rf_loss.jpg")
k = env.get_layer_size()
r = np.random.rand(k)
state = env.get_state()
next_state = state + r
env.step(next_state)