from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def terrain_info(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return terrain information for each environment.
    
    This function returns the terrain level (column index) for each environment, which corresponds
    to the teacher type that should be used for distillation training.
    
    In IsaacLab terrain system:
    - Rows = different terrain types  
    - Cols = difficulty levels (terrain levels)
    - Curriculum increases col index (terrain level), not row index
    
    For multi-teacher distillation:
    - We use terrain level (col index) to select teachers
    - Each teacher specializes in a different difficulty level
    
    Returns:
        torch.Tensor: Terrain levels for each environment (shape: [num_envs, 1])
                     Values range from 0 to (num_cols-1) for teacher selection
    """
    # 检查是否有地形生成器
    if not hasattr(env.scene.terrain, 'terrain_origins') or env.scene.terrain.terrain_origins is None:
        # 如果没有地形原点信息，返回全零（假设在最低难度级别）
        print("Warning: Terrain origins not found. Defaulting to terrain level 0.")
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float)
    
    # 获取地形原点和环境原点
    terrain_origins = torch.tensor(env.scene.terrain.terrain_origins, device=env.device, dtype=torch.float)
    env_origins = env.scene.env_origins
    
    # terrain_origins 的形状是 (num_rows, num_cols, 3)
    # 我们需要将其重塑为 (num_rows * num_cols, 3) 来方便计算距离
    num_rows, num_cols = terrain_origins.shape[:2]
    terrain_origins_flat = terrain_origins.reshape(-1, 3)
    
    # 计算每个环境原点到所有地形原点的距离
    # 只使用 x, y 坐标进行匹配
    distances = torch.cdist(env_origins[:, :2], terrain_origins_flat[:, :2])
    
    # 找到每个环境最近的地形块索引
    closest_terrain_idx = torch.argmin(distances, dim=1)
    
    # 将扁平索引转换为列索引（地形级别/教师ID）
    # 在IsaacLab中，地形级别是沿着列（cols）增加的
    terrain_levels = closest_terrain_idx % num_cols  # 列索引 = 地形级别
    terrain_levels = torch.clamp(terrain_levels, 0, num_cols - 1)
    print("getting terrain levels (teacher IDs):")
    print(terrain_levels)
    return terrain_levels.unsqueeze(-1).float()
    



