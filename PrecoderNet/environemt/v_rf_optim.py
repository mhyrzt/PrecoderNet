import torch as T
import numpy as np
import matplotlib.pyplot as plt

def h(x: T.tensor) -> T.tensor:
    return T.conj(x.transpose(1, 0))


def to_tenser(x: np.ndarray) -> T.tensor:
    return T.tensor(x, dtype=T.cfloat)


def constraint(v_rf: T.tensor, v_bb: T.tensor) -> T.tensor:
    v_rf = v_rf / v_rf.abs()
    t1 = T.matmul(v_rf, v_bb)
    t2 = T.matmul(t1, h(v_bb))
    t3 = T.matmul(t2, h(v_rf))
    tr = T.trace(t3)
    return tr


def compstr(c, ndigits: int = 6):
    r = round(c.real, ndigits)
    i = round(c.imag, ndigits)
    return f"{r} + j{i}"


def find_v_rf(
    v_rf: np.ndarray,
    v_bb: np.ndarray,
    p: float,
    a: float,
    epochs: int
) -> list[np.ndarray]:
    device = "cuda:0" if T.cuda.is_available() else "cpu"
    p -= p / a
    v_bb = to_tenser(v_bb).to(device)
    v_rf = to_tenser(v_rf).to(device).requires_grad_()

    losses = []
    optimizer = T.optim.Adam([v_rf])
    for i in range(epochs):
        optimizer.zero_grad()
        loss = (constraint(v_rf, v_bb) - p) ** 2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    del optimizer
    v_rf = v_rf.detach().cpu().numpy()
    v_rf = v_rf / np.abs(v_rf)
    return v_rf, np.array(losses)


def plot_loss(env):
    v_rf = env.v_rf
    v_bb = env.v_bb
    losses = env.v_rf_loss
    
    x = list(range(len(losses)))
    c = constraint(to_tenser(v_rf), to_tenser(v_bb)).item()
    t = " | ".join(["Loss $ V_{rf} $", f"$ Constraint = {compstr(c)} $"])
    fig, ax = plt.subplots(dpi=200, figsize=(8, 4))
    
    ax.plot(x, np.real(losses), label="$ Real(loss) $")
    ax.plot(x, np.imag(losses), label="$ Imag(loss) $")
    ax.set_title(t)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.legend()
    return fig
