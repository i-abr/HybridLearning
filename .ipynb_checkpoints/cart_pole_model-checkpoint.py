import autograd.numpy as np
from autograd import jacobian
from gym import spaces
    
class CartPoleModel(object):


    def __init__(self):

        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5#0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 1.0 # pole's length
        self.m_p_l = (self.m_p*self.l)
        self.force_mag = 10.0
        self.dt = 0.05  # seconds between state updates
        self.b = 0.1#0.1  # friction coefficient


        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self.fdx = jacobian(self.f, argnum=0)
        self.fdu = jacobian(self.f, argnum=1)

    def set_state(self, x):
        self.state = np.array(x.copy())#, dtype=np.float32)
        return self.state.copy()

    
    def f(self, state, u):
        
        action = np.clip(u, -1.0, 1.0) * self.force_mag

        x, x_dot, theta, theta_dot = state

        s = np.sin(theta)
        c = np.cos(theta)
        
        x = x + x_dot*self.dt
        theta = theta + theta_dot*self.dt
        
        thetadot_update = self.g * s /self.l + action[0] * c/self.l - self.b*theta_dot
        x_dot = x_dot + (action[0] - self.b*x_dot) * self.dt


        theta_dot = theta_dot + thetadot_update*self.dt

        return np.array([x,x_dot,theta,theta_dot])

    @property 
    def A(self):
        return self._A.copy()
    @property
    def B(self):
        return self._B.copy()

    def step(self, action):
        
        state = self.f(self.state, action)

        self._A = self.fdx(self.state, np.array(action))
        self._B = self.fdu(self.state, np.array(action))
        self.state = state
