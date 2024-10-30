import gymnasium as gym
from stable_baselines3 import PPO
import manipulator_mujoco
import time
import argparse

parser = argparse.ArgumentParser(description="Train or Train PPO model on UR5e environment.")
parser.add_argument("--mode", type=str, default="Train", help="Set to 'Train' to train the model or 'Test' to test the model")
args = parser.parse_args()
##############################################################################################################################
################                                             TRAIN                              ##############################
##############################################################################################################################

if args.mode == "Train":
    # Create the environment for training
    env = gym.make('manipulator_mujoco/UR5eEnv-v0', render_mode='human')

    # Reset the environment with a specific seed for reproducibility
    observation, info = env.reset(seed=42)

    # Define the model with PPO algorithm
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model for a specified number of timesteps
    model.learn(total_timesteps=100000)

    # Save the trained model to the current working directory 
    model.save("ppo_ur5e_use_case")

    # Close the environment after training
    env.close()

##############################################################################################################################
################                                             TEST                              ###############################
##############################################################################################################################

elif args.mode == "Test":
    # Load the trained model
    model = PPO.load("ppo_ur5e_use_case") #["Use ppo_ur5e_test to run a stable one"]

    # Re-create the environment after loading the model
    env = gym.make('manipulator_mujoco/UR5eEnv-v0', render_mode='human')
    observation, info = env.reset(seed=42)

    # Run the action loop for testing the model
    for _ in range(10):
        action, _states = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            time.sleep(1)
            if terminated:
                print("TERMINATED ....")
            else:
                print("TRUNCATED ....")
            observation, info = env.reset()

    env.close()
else:
    print("INVALID MODE: Please use 'Train' or 'Test'.")
