import matplotlib.pyplot as plt
import numpy as np
import mujoco
from stable_baselines3 import PPO
from env import AcrobotBalanceEnv

if __name__ == "__main__":
    print("Loading Viewer...")
    env = AcrobotBalanceEnv(render_mode="human")

    print("Loading trained model...")
    model = PPO.load("ppo_acrobot")

    obs, info = env.reset()

    # --- Resolve hinge2 dof index once ---
    hinge2_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge2")
    hinge2_adr = env.model.jnt_dofadr[hinge2_id]

    # --- DATA ARRAYS ---
    t = []
    policy_cmd = []
    ctrl_signal = []
    actuator_force = []
    joint_torque = []
    angle1 =[]
    angle2 =[]

    print("Running simulation and recording data...")

    while True:
        action, _states = model.predict(obs, deterministic=True)

        # Save PPO output (policy command)
        policy_cmd.append(float(action[0]))

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Log real MuJoCo signals AFTER stepping ---
        t.append(float(env.data.time))
        ctrl_signal.append(float(env.data.ctrl[0]))
        actuator_force.append(float(env.data.actuator_force[0]))
        joint_torque.append(float(env.data.qfrc_actuator[hinge2_adr]))
        angle1.append(float(np.degrees(env.data.qpos[0])))
        angle2.append(float(np.degrees(env.data.qpos[1])))

        if terminated or truncated or env.data.time > 20.0:  # Safety stop after 20 seconds
            print(f"Episode finished at {env.data.time:.2f} seconds.")
            break

    env.close()

    #Plotting
    print("Generating Graph...")

    #plt.figure(figsize=(11,5))
    fig, (plt1, plt2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    #plt.plot(t, policy_cmd, label="Policy Output (action)", linewidth=3.0)
    #plt.plot(t, actuator_force, label="Actuator Force", linewidth=1.0)
    plt1.plot(t, angle2, label="top link Angle (deg)",color = 'blue', linewidth=1.5)
    #plt1.plot(t, angle1, label="bottom link Angle (deg)",color = 'red', linewidth=1.5)
    #plt.plot(t, angle1, label="Hinge2 Angle (rad)", linewidth=1.0)
    plt1.axhline(0,color ='black', linestyle="--", linewidth=1)
    #plt1.set_xlabel("Time (seconds)")
    plt1.set_ylabel("Angle (Degrees)")
    plt1.set_title("Global Joint Angles vs Time(s)")
    plt1.legend()
    plt1.grid(True, alpha=0.3)

    plt2.plot(t, joint_torque, label="Joint Torque (Physics)", color='blue', linewidth=1.5)
    plt2.axhline(0, color='black', linestyle="--", linewidth=1)
    plt2.set_xlabel("Time (seconds)")
    plt2.set_ylabel("Torque (Nm)")
    plt2.set_title("Torque Applied to Hinge2 vs Time(s)")
    plt2.legend()
    plt2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()