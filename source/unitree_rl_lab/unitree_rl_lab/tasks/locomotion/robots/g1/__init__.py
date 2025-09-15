import gymnasium as gym

# Unitree G1 29DOF environment
gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Teacher policy environments (Stage 2) - Using teacher_terrain configurations
gym.register(
    id="Isaac-Velocity-Pyramid-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherPyramidEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherPyramidPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Slope-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherSlopeEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherSlopePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Stairs-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherStairsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherStairsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Grid-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherGridEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherGridPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Random-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherRandomEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherRandomPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Wave-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherWaveEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.teacher_terrain:TeacherWavePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)


# Distillation environment (Stage 3 & 4)
gym.register(
    id="Isaac-Velocity-Distillation-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.distillation_env:DistillationEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.distillation_env:DistillationPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_multi_teacher_cfg:MultiTeacherDistillationRunnerCfg",
    },
)
