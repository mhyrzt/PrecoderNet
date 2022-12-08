import numpy as np
from PrecoderNet import H
from .v_rf_optim import find_v_rf


class Environment:
    def __init__(
        self,
        P: float,
        var: float,
        beta: float,
        n_s: int,
        n_t: int,
        n_r: int,
        n_t_rf: int,
        n_r_rf: int,
        n_cl: int,
        n_ray: int,
        channel_matrix: np.ndarray,
    ) -> None:
        self.P: float = P
        self.var: float = var
        self.beta: float = beta  # Channel Imprefection | B = [0.0, 1.0]
        self.I: np.ndarray = np.eye(n_s)

        self.n_s: int = n_s  # Input Signal Len
        self.n_t: int = n_t  # BS Antennas
        self.n_r: int = n_r  # User Antennas
        self.n_t_rf: int = n_t_rf  # BS RF Chains
        self.n_r_rf: int = n_r_rf  # User RF Chains
        self.n_cl: int = n_cl    # Number of Scattering Clusters
        self.n_ray: int = n_ray  # Scattering Rays

        self.size_v_bb: tuple[int] = (self.n_t_rf, self.n_s)
        self.size_v_rf: tuple[int] = (self.n_t_rf, self.n_s)
        self.size_w_rf: tuple[int] = (self.n_r, self.n_r_rf)

        self.channel_matrix: np.ndarray = channel_matrix

        self.w_rf: np.ndarray = self._get_init_w_rf()  # Analog Combiner (Init Random)
        self.v_bb: np.ndarray = self._get_init_v_bb()  # Digital Beamformer (Init Random)
        self.v_rf: np.ndarray = self._calc_v_rf()
        self.w_bb: np.ndarray = self._calc_w_bb()


    def _rand(self, size: tuple, low: float, high: float, j: bool = False) -> np.ndarray:
        """_summary_

        Args:
            size (tuple): random array size
            low (float): minimum value
            high (float): maximum value
            j (bool, optional): imaginary. Defaults to False.

        Returns:
            np.ndarray: random array
        """
        c = 1.j if j else 1.0
        r = np.random.uniform(low, high, size)
        return r * c

    def _complex_rand(self, size: tuple, low: float = -1, high: float = 1) -> np.ndarray:
        """_summary_

        Args:
            size (tuple): random complex array size
            low (float, optional): minimum value. Defaults to -1.
            high (float, optional): maximum. Defaults to 1.

        Returns:
            np.ndarray: random complex array
        """
        r = self._rand(size, low, high, False)
        i = self._rand(size, low, high, True)
        return r + i

    def _get_init_v_bb(self) -> np.ndarray:
        return self._complex_rand(self.size_v_bb)

    def _get_init_w_rf(self) -> np.ndarray:
        w_rf = self._complex_rand(self.size_w_rf)
        w_rf = w_rf / np.abs(w_rf)
        return w_rf

    def _reshape(self, real: np.ndarray, imag: np.ndarray, size: tuple) -> np.ndarray:
        return np.reshape(real + imag,  size)

    def get_layer_size(self) -> int:
        k = 2 * (self.n_t_rf * self.n_s + self.n_r * self.n_r_rf)
        return k

    def _reward(self) -> float:
        # Formula (6)
        w = self._calc_w_t()
        v = self._calc_v_t()
        c = self._calc_c_n()
        h = self.channel_matrix
        c_1 = np.linalg.inv(c)

        A = (1 + (self.beta * self.P / self.var)) * np.eye(self.n_s)
        B = (1 - self.beta) * c_1 @ H(w) @ h @ v @ H(v) @ H(h) @ w
        print("reward", "B", B, B.shape)
        return np.log2(np.linalg.det(A + B))

    def step(self, action: np.ndarray) -> list[np.ndarray, float]:
        # To do
        return self.get_state(), self._reward()

    def _calc_v_rf(self) -> np.ndarray:
        init = self._complex_rand(self.size_v_rf)
        v_rf, losses = find_v_rf(init, self.v_bb, self.P, 10, 10_000)
        self.v_rf_loss = losses
        return v_rf

    def _calc_v_t(self):
        return self.v_rf @ self.v_bb

    def _calc_w_t(self) -> np.ndarray:
        return self.w_rf @ self.w_bb

    def _calc_c_n(self) -> np.ndarray:
        return self.var * H(self.w_bb) @ H(self.w_rf) @ self.w_rf @ self.w_bb

    def _calc_psi(self) -> np.ndarray:
        h = self.channel_matrix
        v_t = self._calc_v_t()

        a = (1 - self.beta)
        b = h @ H(v_t) @ H(h)
        c = (self.beta * self.P + self.var) * np.eye(self.n_r)

        return a * b + c

    def _calc_w_bb(self) -> np.ndarray:
        # Formula (8)s
        h = self.channel_matrix
        v_t = self._calc_v_t()
        psi = self._calc_psi()
        cof = np.sqrt(1 - self.beta)
        w_rf = self.w_rf
        w_rf_h = H(w_rf)
        w_bb = cof * (np.linalg.inv(w_rf_h @ psi @ w_rf) @ w_rf_h @ h @ v_t)
        return w_bb

    def _flat_real_imag(self, arr: np.ndarray) -> list[np.ndarray]:
        f = arr.flatten()
        return f.real, f.imag

    def get_state(self) -> np.ndarray:
        flat_v_bb = self._flat_real_imag(self.v_bb)
        flat_w_rf = self._flat_real_imag(self.w_rf)
        return np.concatenate([*flat_v_bb, *flat_w_rf])
