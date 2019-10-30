import numpy as np
from gym import spaces

class VDP(object):

    def __init__(self):
        self.action_space       = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space  = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        #self.state = np.random.uniform(-4., 4., size=self.observation_space.shape)
        self.state = np.random.uniform(-1., 1., size=self.observation_space.shape)
        return self.state.copy()

    def get_reward(self):
        return np.dot(self.state, self.state)

    def step(self, u): ### van der pol oscillator dynamic step
        a = np.clip(u, self.action_space.low, self.action_space.high)
        reward = self.get_reward()

        x = self.state
        #dx = np.array([x[1], -x[0] +  1.0 * (1-x[0]**2)*x[1] + a[0]])
        dx = np.array([2.0 * x[1], -0.8*x[0] + 2.0 * x[1] - 10.0 * (x[0]**2) * x[1] + a[0]])
        self.state = self.state + 0.05 * dx

        return self.state.copy(), -reward, False, {}
