import torch as T
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


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

    p -= p / a
    v_bb = to_tenser(v_bb)
    v_rf = to_tenser(v_rf).requires_grad_()

    losses = []
    optimizer = T.optim.Adam([v_rf])
    for i in (pbar := tqdm(range(epochs), ncols=100)):
        optimizer.zero_grad()
        loss = (constraint(v_rf, v_bb) - p) ** 2
        if i % 50 == 0:
            pbar.set_description(f"v_rf loss = {compstr(loss.item())}")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    del optimizer
    v_rf = v_rf.detach().cpu().numpy()
    v_rf = v_rf / np.abs(v_rf)
    return v_rf, np.array(losses)


def plot_loss(v_rf: np.ndarray, v_bb: np.ndarray, losses: np.ndarray, start: int=0):
    assert start >= 0 and start < len(losses), "start should be gte 0 and lt len(loss)"
    y = losses[start:]
    x = list(range(len(y)))
    c = constraint(to_tenser(v_rf), to_tenser(v_bb)).item()
    t = " | ".join(["Loss $ V_{rf} $", f"$ Constraint = {compstr(c)} $"])
    fig, ax = plt.subplots(dpi=200, figsize=(8, 4))
    
    ax.plot(x, np.real(y), label="$ Real(loss) $")
    ax.plot(x, np.imag(y), label="$ Imag(loss) $")
    ax.set_title(t)
    ticks = ax.get_xticks() + start
    ticks = ticks.astype(np.int32)
    ax.set_xticklabels(ticks)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.legend()
    return fig
