import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController

class UR5eEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    def __init__(self, render_mode=None):
        # Observation space includes joint positions, velocities, and the target distance
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64  
        )

        # Action space to control joint positions directly (6 joints)
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(6,), dtype=np.float64  
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################
        self._arena = StandardArena()
        self._target = Target(self._arena.mjcf_model)
        self._arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )

        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,0], quat=[0.7071068, 0, 0, -0.7071068]
        )
        
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        # Time-related variables
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

    def _get_obs(self) -> np.ndarray:
        # Observations include joint positions, velocities, and the distance to the target
        joint_positions = self._physics.bind(self._arm.joints).qpos
        joint_velocities = self._physics.bind(self._arm.joints).qvel
        ee_pos = self._physics.bind(self._arm.eef_site).xpos
        
        # Fix: Assuming get_mocap_pose returns an array-like object with [x, y, z]
        target_pos = self._target.get_mocap_pose(self._physics)[:3]
        
        distance_to_target = np.linalg.norm(ee_pos - target_pos)
        
        return np.concatenate([joint_positions, joint_velocities, [distance_to_target]])

    def _get_info(self) -> dict:
        # Add info about end-effector and target positions
        ee_pos = self._physics.bind(self._arm.eef_site).xpos
        
        # Fix: Assuming get_mocap_pose returns an array-like object with [x, y, z]
        target_pos = self._target.get_mocap_pose(self._physics)[:3]

        return {"ee_position": ee_pos, "target_position": target_pos}

    ## uncomment this to have static target position for training
    # def reset(self, seed=None, options=None) -> tuple:
    #     super().reset(seed=seed)

    #     # Reset physics and set arm to a default starting position
    #     with self._physics.reset_context():
    #         self._physics.bind(self._arm.joints).qpos = [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]
    #         self._target.set_mocap_pose(self._physics, position=[0.5, 0, 0.3], quaternion=[0, 0, 0, 1])

    #     observation = self._get_obs()
    #     info = self._get_info()
    #     return observation, info

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        # Reset physics and set arm to a default starting position
        with self._physics.reset_context():
            self._physics.bind(self._arm.joints).qpos = [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]
            
            # Generate random target positions and orientations
            target_position = np.random.uniform(low=[0.4, -0.2, 0.2], high=[0.6, 0.2, 0.4])
            target_quaternion = [0, 0, 0, 1]  # Keep orientation fixed, or generate randomly if needed

            # Set mocap pose to new target
            self._target.set_mocap_pose(self._physics, position=target_position, quaternion=target_quaternion)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info


    def step(self, action: np.ndarray) -> tuple:
        # Apply the action by modifying joint positions
        joint_positions = self._physics.bind(self._arm.joints).qpos
        new_joint_positions = joint_positions + action

        # Ensure that the joint positions are valid
        self._physics.bind(self._arm.joints).qpos = np.clip(new_joint_positions, -np.pi, np.pi)

        # Run the controller to move the arm based on the new joint positions
        target_pose = self._target.get_mocap_pose(self._physics)
        self._controller.run(target_pose)

        # Step the simulation
        self._physics.step()

        # Get new observations
        observation = self._get_obs()
        reward, terminated = self._compute_reward_and_done(observation)

        # Render if needed
        if self._render_mode == "human":
            self._render_frame()


        info = self._get_info()

        if terminated:
            print("EPISODE TERMINATED")
            # time.sleep(5)

        return observation, reward, terminated, False, info

    def _compute_reward_and_done(self, observation: np.ndarray) -> tuple:
        # Distance between end-effector and target
        distance_to_target = observation[-1]

        # Simple reward: negative of the distance (closer is better)
        reward = -distance_to_target

        # Terminate the episode if the distance to the target is small enough
        terminated = distance_to_target < 0.05  # Task success threshold

        return reward, terminated

    def render(self) -> np.ndarray:
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        if self._viewer is None and self._render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            self._step_start = time.time()

        if self._render_mode == "human":
            self._viewer.sync()

            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()
        else:
            return self._physics.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
