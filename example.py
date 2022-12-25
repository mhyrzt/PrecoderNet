from config import *
from PrecoderNet import Trainer
from PrecoderNet.ddpg import DDPG
from PrecoderNet.models import Actor, Critic
from PrecoderNet.environemt import Environment, plot_loss
from PrecoderNet.random_process import OrnsteinUhlenbeckProcess

env = Environment(**ENV_CONFIG)
plot_loss(env).savefig("results/v_rf_loss.jpg")
k = env.get_layer_size()
random_process = OrnsteinUhlenbeckProcess(
    size=k,
    theta=0.15,
    mu=0.0,
    sigma=0.2
)
ddpg = DDPG(
    Actor(k, k, (256, 256, 256)),
    Critic(k, k, (256, 256, 256)),
    MEM_MAX_LEN,
    MEM_BATCH_SIZE,
    random_process
)

Trainer(env, ddpg, EPOCHS) \
    .train() \
    .save_progress_plot(RESULT_FOLDER)
