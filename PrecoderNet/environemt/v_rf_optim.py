import torch as T
import numpy as np
import matplotlib.pyplot as plt


def h(x: T.tensor) -> T.tensor:
    return T.conj(x.transpose(1, 0))


def to_tenser(x: np.ndarray) -> T.tensor:
    return T.tensor(x, dtype=T.cfloat)


def constraint(v_rf: T.tensor, v_bb: T.tensor) -> T.tensor:
    t1 = T.matmul(v_rf, v_bb)
    t2 = T.matmul(t1, h(v_bb))
    t3 = T.matmul(t2, h(v_rf))
    tr = T.trace(t3)
    return tr


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
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = (constraint(v_rf, v_bb) - p) ** 2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return v_rf.detach().cpu().numpy(), np.array(losses)


def plot_loss(v_rf, v_bb, err, start=0):
    x = list(range(len(y)))
    y = err[-start:]
    c = constraint(v_rf, v_bb).item()
    r = round(c.real, 4)
    i = round(c.imag, 4)
    l = len(err) - start
    t = f"LOSS $ v_rf $ | $ Constraint = {r} + j{i} $ | last {l}"
    
    fig, ax = plt.subplots(dpi=200)
    ax.plot(x, np.real(y), label="$ Real(loss) $")
    ax.plot(x, np.imag(y), label="$ Imag(loss) $")
    ax.set_title(t)
    ax.grid()
    ax.legend()
    return fig
