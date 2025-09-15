# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg


@configclass
class MultiTeacherStudentCfg:
    """Configuration for MultiTeacherStudent policy."""
    class_name: str = "MultiTeacherStudent"
    num_teachers: int = 6
    teacher_model_paths: Optional[List[str]] = None
    
    # Network configuration
    init_noise_std: float = 1.0
    activation: str = "elu"
    noise_std_type: str = "scalar"
    
    # Student network
    student_obs_normalization: bool = True
    student_hidden_dims: List[int] = [512, 256, 128]
    
    # Teacher network
    teacher_obs_normalization: bool = True
    teacher_hidden_dims: List[int] = [512, 256, 128]


@configclass
class MultiTeacherDistillationCfg:
    """Configuration for MultiTeacherDistillation algorithm."""
    class_name: str = "MultiTeacherDistillation"
    teacher_model_paths: Optional[List[str]] = None
    
    # Training hyperparameters
    num_learning_epochs: int = 1
    gradient_length: int = 15
    learning_rate: float = 5.0e-4
    max_grad_norm: float = 1.0
    loss_type: str = "mse"
    optimizer: str = "adam"


@configclass
class MultiTeacherDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Multi-teacher distillation training configuration."""
    experiment_name: str = "multi_teacher_distillation"
    run_name: str = "multi_teacher_dist"
    num_steps_per_env: int = 24
    save_interval: int = 200
    
    policy: MultiTeacherStudentCfg = MultiTeacherStudentCfg()
    algorithm: MultiTeacherDistillationCfg = MultiTeacherDistillationCfg()
    obs_groups: dict = {
        "policy": ["policy"],
        "teacher": ["teacher"],
        "terrain_info": ["terrain_info"]
    }


def get_multi_teacher_distillation_cfg(teacher_model_paths: List[str]) -> MultiTeacherDistillationRunnerCfg:
    """Get multi-teacher distillation configuration with teacher model paths.
    
    Args:
        teacher_model_paths: List of 6 teacher model checkpoint paths
                           Order: [pyramid, slope, stairs, grid, random, wave]
    
    Returns:
        MultiTeacherDistillationRunnerCfg: Complete configuration
    """
    if len(teacher_model_paths) != 6:
        raise ValueError(f"Expected 6 teacher model paths, got {len(teacher_model_paths)}")
    
    cfg = MultiTeacherDistillationRunnerCfg()
    
    # Set teacher model paths
    cfg.policy.teacher_model_paths = teacher_model_paths
    cfg.algorithm.teacher_model_paths = teacher_model_paths
    
    return cfg
