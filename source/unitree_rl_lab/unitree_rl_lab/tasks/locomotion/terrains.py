"""
Terrain configurations for teacher policy training.
This module defines various terrain types for multi-terrain teacher training.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

# 1. Pyramid terrain configuration
PYRAMID_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,  # Increased for more environments
    num_cols=8,  # Increased for more environments
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

# 2. Slope terrain configuration
SLOPE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=32,  # Increased for more environments
    num_cols=32,  # Increased for more environments
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

# 3. Stairs terrain configuration
STAIRS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=32,  # Increased for more environments
    num_cols=32,  # Increased for more environments
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

# 4. Grid/Checkerboard terrain configuration
GRID_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=32,  # Increased for more environments
    num_cols=32,  # Increased for more environments
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.5, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
    },
)

# 5. Random rough terrain configuration
RANDOM_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=32,  # Increased for more environments
    num_cols=32,  # Increased for more environments
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "hf_pyramid_discrete": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            obstacle_height_mode="choice", 
            obstacle_height_range=(0.05, 0.15),
            obstacle_width_range=(0.2, 0.8),  # Added missing parameter
            num_obstacles=10,  # Added missing parameter
            platform_width=2.0, 
            border_width=0.25
        ),
    },
)

# 6. Triangle mesh/wave terrain configuration
WAVE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=32,  # Increased for more environments
    num_cols=32,  # Increased for more environments
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),  # Increased for robot initialization
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.01, 0.08),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)

# List of all terrain configurations for easy iteration
TEACHER_TERRAIN_CONFIGS = [
    ("wave", WAVE_TERRAIN_CFG),
    ("pyramid", PYRAMID_TERRAIN_CFG),
    ("stairs", STAIRS_TERRAIN_CFG),
    ("grid", GRID_TERRAIN_CFG),
    ("random", RANDOM_TERRAIN_CFG),
    ("slope", SLOPE_TERRAIN_CFG), 
]
