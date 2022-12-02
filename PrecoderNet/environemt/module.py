import numpy as np
from PrecoderNet import H


class Environment:
    def __init__(
        self,
        n_s: int,
        n_t: int,
        n_r: int,
        n_t_rf: int,
        n_r_rf: int,
        n_cl: int,
        n_ray: int,
        beta: float,
        channel_matrix: np.ndarray,
        P: float,
        var: float
    ) -> None:

        self.P: float = P
        self.var: float = var
        self.beta: float = beta  # Channel Imprefection | B = [0.0, 1.0]

        self.n_s: int = n_s  # Input Signal Len
        self.n_t: int = n_t  # BS Antennas
        self.n_r: int = n_r  # User Antennas
        self.n_t_rf: int = n_t_rf  # BS RF Chains
        self.n_r_rf: int = n_r_rf  # User RF Chains
        self.n_cl: int = n_cl    # Number of Scattering Clusters
        self.n_ray: int = n_ray  # Scattering Rays

        self.size_v_bb: tuple[int] = (self.n_t_rf, self.n_s)
        self.size_w_rf: tuple[int] = (self.n_r, self.n_r_rf)

        self.w_rf: np.ndarray = self._get_init_w_rf()  # Analog Combiner
        self.v_bb: np.ndarray = self._get_init_v_bb()  # Digital Beamformer
        self.v_rf: np.ndarray = self._calc_v_rf()
        self.w_bb: np.ndarray = self._calc_w_bb()

        self.channel_matrix: np.ndarray = channel_matrix

    def _rand(self, size: tuple, low: float, high: float, j: bool = False):
        c = 1.j if j else 1.0
        r = np.random.uniform(low, high, size)
        return r * c

    def _complex_rand(self, size: tuple, low: float = -1, high: float = 1) -> np.ndarray:
        r = self._rand(size, low, high, False)
        i = self._rand(size, low, high, True)
        return r + i

    def _get_init_v_bb(self) -> np.ndarray:
        return self._complex_rand(self.size_v_bb)

    def _get_init_w_rf(self) -> np.ndarray:
        return self._complex_rand(self.size_w_rf)

    def _reshape(self, real: np.ndarray, imag: np.ndarray, size: tuple) -> np.ndarray:
        return np.reshape(real + imag,  size)

    def get_layer_size(self) -> int:
        k = 2 * (self.n_t_rf * self.n_s + self.n_r * self.n_r_rf)
        return k

    def _reward(self):
        # Formula (6)
        w = self._calc_w_t()
        v = self._calc_v_t()
        c = self._calc_c_n()
        h = self.channel_matrix
        
        b2 = self.beta ** 2
        c_1 = np.linalg.inv(c)
        
        A = (1 + (b2 * self.P / self.var)) * np.eye(self.n_s)
        B = (1 - b2) * c_1 @ H(w) @ h @ v @ H(v) @ H(h) @ w
        return np.log2(np.linalg.det(A + B))

    def step(self, action: np.ndarray):
        return

    def noise(self):
        k = self.get_layer_size()
        return

    def _calc_v_rf(self):
        pass

    def _calc_v_t(self):
        return self.v_rf @ self.v_bb

    def _calc_w_t(self) -> np.ndarray:
        return self.w_rf @ self.w_bb

    def _calc_c_n(self) -> np.ndarray:
        return self.var * H(self.w_bb) @ H(self.w_rf) @ self.w_rf @ self.w_bb

    def _calc_psi(self) -> np.ndarray:
        h = self.channel_matrix
        v_t = self._calc_v_t()
        b_2 = self.beta ** 2

        a = (1 - b_2)
        b = h @ H(v_t) @ H(h)
        c = (b_2 * self.P + self.var) * np.eye(self.n_r)

        return a * b + c

    def _calc_w_bb(self):
        # Formula (8)s
        h = self.channel_matrix
        v_t = self._calc_v_t()
        psi = self._calc_psi()
        cof = np.sqrt(1 - (self.beta ** 2))
        w_rf = self.w_rf
        w_rf_h = H(w_rf)

        return cof * (np.linalg.inv(w_rf_h @ psi @ w_rf) @ w_rf_h @ h @ v_t)
