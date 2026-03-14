from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
from env import AcrobotBalanceEnv
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    print("Setting up Acrobot Environment...")
    
    # 6 parallel environments for the 6 physical cores
    
    
    single_env = AcrobotBalanceEnv(render_mode=None)
    env = TimeLimit(single_env, max_episode_steps=1000)
    check_env(env)
    vec_env = make_vec_env(
        AcrobotBalanceEnv, 
        n_envs=6, 
        wrapper_class=TimeLimit, 
        wrapper_kwargs={"max_episode_steps": 1000}
    )
    
    # We use the [256, 256] network because Acrobot kinematics are highly non-linear
    policy_kwargs = dict(net_arch=[256, 256])
    
    # Calculate appropriate sizes for 6 environments
    # 512 steps * 6 envs = 3072 total rollout buffer (much closer to the standard 2048)
    
    model = PPO("MlpPolicy", vec_env, verbose=1, 
                policy_kwargs=policy_kwargs,
                n_steps=512,           # Lowered from default 2048
                batch_size=256,        # Explicitly set batch size for clean mini-batches
                tensorboard_log="./acrobot_tb/")
    
    print("Starting training...")
    # 1.5 Million steps should be plenty for upright balancing
    model.learn(total_timesteps=300_000, tb_log_name="ppo_acrobot_balance")
    
    model.save("ppo_acrobot")
    print("Training finished!")