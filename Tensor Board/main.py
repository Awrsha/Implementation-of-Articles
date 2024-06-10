import argparse
import gym
from stable_baselines3 import A2C, TD3, SAC
import os

# Directories for saving models and logs
log_dir = "logs"
model_dir = "models"

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env, algorithm, timesteps):
    """
    Function to train a reinforcement learning model using the specified algorithm.

    Args:
        env (str): Name of the Gym environment.
        algorithm (str): Name of the algorithm to use (A2C, TD3, SAC).
        timesteps (int): Number of training steps.

    Returns:
        None
    """
    # Select the appropriate algorithm
    if algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    else:
        print("Algorithm is not supported.")
        exit(1)
    
    i = 0
    while True:
        i += 1
        # Train the model for the specified number of timesteps
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        # Save the model
        model.save(f"{model_dir}/{algorithm}_{timesteps * i}")

def test(env, algorithm, model_path):
    """
    Function to test a pre-trained reinforcement learning model.

    Args:
        env (str): Name of the Gym environment.
        algorithm (str): Name of the algorithm used for training.
        model_path (str): Path to the pre-trained model.

    Returns:
        None
    """
    # Load the pre-trained model
    if algorithm == 'A2C':
        model = A2C.load(model_path, env=env)
    elif algorithm == 'TD3':
        model = TD3.load(model_path, env=env)
    elif algorithm == 'SAC':
        model = SAC.load(model_path, env=env)
    else:
        print("Algorithm not found.")
        exit(1)
    
    # Run the model in the environment and render the output
    obs = env.reset()
    steps = 500
    while steps > 0:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            steps -= 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test a reinforcement learning model using Stable Baselines3 and Gym environment")
    parser.add_argument("gymenv", help="Name of the Gym environment (e.g., Humanoid-v4, CartPole-v1)")
    parser.add_argument("algorithm", help="Choose an algorithm: A2C, TD3, SAC")
    parser.add_argument("-t", "--train", action='store_true', help="Train the model")
    parser.add_argument("-e", "--test", metavar='model_path', help="Test a pre-trained model")
    parser.add_argument("-s", "--steps", type=int, default=20000, help="Number of training steps (default: 20000)")
    args = parser.parse_args()

    if args.train:
        print("Training started...")
        # Create the Gym environment
        gymenv = gym.make(args.gymenv)
        # Train the model
        train(gymenv, args.algorithm, args.steps)
        print("Training completed.")
    
    if args.test:
        print("Testing started...")
        # Create the Gym environment
        gymenv = gym.make(args.gymenv)
        # Test the pre-trained model
        test(gymenv, args.algorithm, model_path=args.test)
        print("Testing completed.")
