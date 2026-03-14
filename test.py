from stable_baselines3 import PPO
from env import AcrobotBalanceEnv

if __name__ == "__main__":
    # 1. Initialize environment with human rendering turned on
    print("Loading Viewer...")
    env = AcrobotBalanceEnv(render_mode="human")
    
    # 2. Load the trained brain
    print("Loading trained model...")
    model = PPO.load("ppo_acrobot")
    
    # 3. Start the visual loop
    obs, info = env.reset()
    
    while True:
        # Get the AI's action based on the current observation
        action, _states = model.predict(obs, deterministic=True)
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # If the pendulum falls or hits the edge, reset and try again
        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, info = env.reset()