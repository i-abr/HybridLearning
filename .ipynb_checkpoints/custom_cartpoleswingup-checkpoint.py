"""
Cart pole swing-up: Original version from:
https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py

Modified so that done=True when x is outside of -2.4 to 2.4
Reward is also reshaped to be similar to PyBullet/roboschool version

More difficult, since dt is 0.05 (not 0.01), and only 200 timesteps
"""

import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import cos, sin

logger = logging.getLogger(__name__)

def wrap2pi(x):
    x = np.fmod(x + np.pi, 2.0 * np.pi)
    if x<0:
        x += 2.0 * np.pi
    return x-np.pi

class Objective(object):
    """
    Contains the task specifications for the VDP system
    """
    def __init__(self):
        pass
    def l(self, x, u):
        """ Going to assume that the state is...
            x, xdot, theta, thetadot 
        """
        c = np.cos(x[2])
        s = np.sin(x[2])
        return 100.*(x[0]**2+(1.-c)**2)+0.1*(s**2+x[3]**2+x[1]**2+u[0]**2)

    def m(self, x):
        return 0.0
    """
    Here we define the derivatives for the task 
    """

    def ldx(self, x_t):
        theta, x, theta_dot, x_dot = x_t.copy()
        theta = wrap2pi(theta)
        ldx = np.zeros(x_t.shape)
        ldx[0] = 2.0 * 200. * theta
        ldx[1] = 10. * x **3
        ldx[2] = 0.1*theta_dot
        ldx[3] = 2.0*50. * x_dot
        return ldx

    def ldu(self, u):
        return np.array([0.2*u[0]])

class CartPoleSwingUpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.dt = 0.02
        self.g = 9.81
        self.l = 1.0
        
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,))
        self.task = Objective()
        self.viewer = None
        self.x_threshold = 2.4

    @property
    def A(self):
        return self._A.copy()
    @property
    def B(self):
        return self._B.copy()
        
    def fdx(self, x, u):
        A = np.array([
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [self.g*cos(x[0])/self.l-u[0]*sin(x[0])/self.l, 0., -0.2, 0.],
            [0., 0., -0.1, 0.]
        ])
        return np.eye(4) + A*self.dt
    def fdu(self, x, u):
        B = np.array([
            [0.],
            [0.],
            [cos(x[0])/self.l],
            [1.]
        ])
        return B*self.dt
    
    def step(self, u):
        theta, x, theta_dot, x_dot = self.state
        xdot = np.array([
            theta_dot,
            x_dot,
            self.g*sin(theta)/self.l + u[0]*cos(theta)/self.l - 0.2*theta_dot,
            u[0] - 0.1*x_dot
        ])
        self._A = self.fdx(self.state, u)
        self._B = self.fdu(self.state, u)
        self.state = self.state + xdot*self.dt
        return self.state.copy()
    
    def set_state(self, x):
        self.state = np.array(x.copy())#, dtype=np.float32)
        return self.state.copy()
    
    def reset(self, x=None):
        
        if x is None:
            self.state = np.random.normal(loc=np.array([np.pi, 0.0, 0.0, 0.0]), scale=0.2)
        else:
            self.state = np.array(x.copy())
        return self.state.copy()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600 # before was 400

        world_width = 5  # max visible position of cart
        scale = screen_width/world_width
        carty = screen_height/2 # TOP OF CART
        polewidth = 6.0
        polelen = scale*self.l  # 0.6 or self.l
        cartwidth = 40.0
        cartheight = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2

            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)

            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            # Make another circle on the top of the pole
            self.pole_bob = rendering.make_circle(polewidth/2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cartheight/4)
            self.wheel_r = rendering.make_circle(cartheight/4)
            self.wheeltrans_l = rendering.Transform(translation=(-cartwidth/2, -cartheight/2))
            self.wheeltrans_r = rendering.Transform(translation=(cartwidth/2, -cartheight/2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
            self.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            self.track = rendering.Line((screen_width/2 - self.x_threshold*scale,carty - cartheight/2 - cartheight/4),
              (screen_width/2 + self.x_threshold*scale,carty - cartheight/2 - cartheight/4))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[1]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[0])
        self.pole_bob_trans.set_translation(-self.l*np.sin(x[0]), self.l*np.cos(x[0]))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
