import numpy as np


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
        wave_len: float,
        phi: float,
        d: float,
        v_rf_thresh: float
    ) -> None:
        self.n_s: int = n_s  # Input Signal Len ?!
        self.n_t: int = n_t  # BS Antennas
        self.n_r: int = n_r  # User Antennas
        self.n_t_rf: int = n_t_rf  # BS RF Chains
        self.n_r_rf: int = n_r_rf  # User RF Chains

        self.n_cl: int = n_cl   # Number of Scattering Clusters
        self.n_ray: int = n_ray  # Scattering Rays

        self.d: float = d       # Antenna Spacing
        self.phi: float = phi   # AoA/Aod
        self.wave_len: float = wave_len
        self.v_rf_thresh: float = v_rf_thresh

        self.w_rf: np.ndarray = self._get_init_w_rf() # Analog Combiner
        self.v_bb: np.ndarray = self._get_init_v_bb() # Digital Beamformer

    def _rand(self, size: tuple, low: float, high: float, j: bool = False):
        c = 1.j if j else 1.0
        r = np.random.uniform(low, high, size)
        return  r * c

    def _complex_rand(self, size: tuple, low: float = -1, high: float = 1) -> np.ndarray:
        r = self._rand(size, low, high, False)
        i = self._rand(size, low, high, True)
        return r + i

    def _get_init_v_bb(self) -> np.ndarray:
        size = (self.n_t_rf, self.n_s)
        return self._complex_rand(size)

    def _get_init_w_rf(self) -> np.ndarray:
        size = (self.n_r, self.n_r_rf)
        return self._complex_rand(size)

    def get_w_bb(self):
        # Formula (8)
        return

    def get_layer_size(self) -> int:
        k = 2 * (self.n_t_rf * self.n_s + self.n_r * self.n_r_rf)
        return k

    def _reward(self):
        # Formula (9)
        return

    def step(action):
        return
    
    def noise(self):
        k = self.get_layer_size()
        return
        
        
