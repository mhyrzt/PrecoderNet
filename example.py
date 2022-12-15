from config import CONFIG
from PrecoderNet.environemt import Environment, plot_loss
from PrecoderNet.models import Actor, Critic
import numpy as np
env = Environment(**CONFIG)
plot_loss(env.v_rf, env.v_bb, env.v_rf_loss).savefig("v_rf_loss.jpg")
k = env.get_layer_size()

actor = Actor(k, k, (512, 512, 512))
critic = Actor(k, k, (512, 512, 512))
print(actor)
print(critic)