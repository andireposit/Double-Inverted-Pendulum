import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time

ACROBOT_XML = """
<mujoco>
  <compiler angle="radian"/>
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
  </visual>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba=".8 .9 .8 1"/>
    
    <body name="base" pos="0 0 1.2">
      <geom type="cylinder" size="0.05 0.1" rgba="0.2 0.2 0.2 1" axisangle="0 1 0 1.57"/>
      
      <body name="pole1" pos="0 0 0">
        <joint name="hinge1" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom name="pole1_geom" type="capsule" fromto="0 0 0 0 0 0.5" size="0.04" rgba="0 0.3 0.7 1"/>
        
        <body name="pole2" pos="0 0 0.5">
          <joint name="hinge2" type="hinge" axis="0 1 0" damping="0.01"/>
          <geom name="pole2_geom" type="capsule" fromto="0 0 0 0 0 0.5" size="0.03" rgba="0 0.7 0.3 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor joint="hinge2" name="middle_motor" gear="5"/>
  </actuator>
</mujoco>
"""

class AcrobotBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(ACROBOT_XML)
        self.data = mujoco.MjData(self.model)
        
        # Action Space: 1 Motor (Torque applied to the middle hinge)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space: 4 variables -> [pole1_angle, pole2_angle, pole1_vel, pole2_vel]
        # (Since the cart is gone, the observation space is smaller and more focused)
        high = np.array([1.0, 1.0, 1.0, 1.0,np.inf,np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        self.viewer = None

    def _get_obs(self):
        # Mandatory float32 conversion for Stable Baselines3 on your laptop
        theta1 = self.data.qpos[0]
        theta2 = self.data.qpos[1]
        vel1 = self.data.qvel[0]
        vel2 = self.data.qvel[1]

        obs = np.array([np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2), vel1, vel2], dtype=np.float32)
        return obs 
        #return np.concatenate([self.data.qpos, self.data.qvel]).ravel().astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start perfectly upright with microscopic noise
        # qpos[0] = hinge1, qpos[1] = hinge2
        self.data.qpos[0] += np.random.uniform(-0.1, 0.1)
        self.data.qpos[1] = 0.4
        #self.data.qpos[1] += np.random.uniform(-0.1, 0.1)
        
        mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[0] = action[0]

        # Frame skip of 10 (0.02s per step) allows the momentum to transfer
        # from the actuated middle joint down to the passive base joint.
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        cos_theta1, cos_theta2 = obs[0], obs[2]
        
        # --- BALANCING REWARD ---
        # +10 points for surviving
        reward = (cos_theta1-0.825) + (cos_theta2-0.825)  # Reward is higher when both poles are closer to upright (cosine near 1)
        
        # Severe penalty for leaning. The agent must keep both hinges near 0.
        

        # Penalty for aggressive motor usage (encourages smooth balance)
        reward -= 0.01 * (action[0]**2)

        # Fail Condition: If either pole tilts more than ~34 degrees (0.6 radians)
        terminated = bool(
            cos_theta1 < 0.825 or cos_theta2 < 0.825  # If either pole is not upright (cosine < 0.825)

        )

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)

    def close(self):
        if self.viewer:
            self.viewer.close()