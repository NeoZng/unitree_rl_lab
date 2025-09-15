#!/bin/bash

# Multi-Teacher Distillation Training Example
# 请根据实际的教师模型路径修改下面的路径

echo "🚀 Starting Multi-Teacher Distillation Training for Unitree G1..."

# 教师模型路径（请根据实际情况修改）
TEACHER_PYRAMID="C:\\Users\\BrainCo\\Desktop\\unitree_rl_lab\\logs\\rsl_rl\\unitree_g1_29dof_velocity\\2025-09-15_11-59-13\\model_3600.pt"
TEACHER_SLOPE="C:\\Users\\BrainCo\\Desktop\\unitree_rl_lab\\logs\\rsl_rl\\unitree_g1_29dof_velocity\\2025-09-15_11-59-13\\model_3600.pt"
TEACHER_STAIRS="C:\\Users\\BrainCo\\Desktop\\unitree_rl_lab\\logs\\rsl_rl\\unitree_g1_29dof_velocity\\2025-09-15_11-59-13\\model_3600.pt"
TEACHER_GRID="C:\\Users\\BrainCo\\Desktop\\unitree_rl_lab\\logs\\rsl_rl\\unitree_g1_29dof_velocity\\2025-09-15_11-59-13\\model_3600.pt"
TEACHER_RANDOM="C:\\Users\\BrainCo\\Desktop\\unitree_rl_lab\\logs\\rsl_rl\\unitree_g1_29dof_velocity\\2025-09-15_11-59-13\\model_3600.pt"
TEACHER_WAVE="C:\\Users\\BrainCo\\Desktop\\unitree_rl_lab\\logs\\rsl_rl\\unitree_g1_29dof_velocity\\2025-09-15_11-59-13\\model_3600.pt"

# 检查教师模型是否存在
echo "🔍 Checking teacher models..."
python scripts/rsl_rl/train_multi_teacher.py \
    --teacher_check \
    --teacher_paths \
        $TEACHER_PYRAMID \
        $TEACHER_SLOPE \
        $TEACHER_STAIRS \
        $TEACHER_GRID \
        $TEACHER_RANDOM \
        $TEACHER_WAVE

# 如果检查通过，开始训练
if [ $? -eq 0 ]; then
    echo "✅ All teacher models validated. Starting training..."
    
    python scripts/rsl_rl/train_multi_teacher.py \
        --task Isaac-UnitreeG1DistillationEnv-v0 \
        --teacher_paths \
            $TEACHER_PYRAMID \
            $TEACHER_SLOPE \
            $TEACHER_STAIRS \
            $TEACHER_GRID \
            $TEACHER_RANDOM \
            $TEACHER_WAVE \
        --num_envs 512 \
        --max_iterations 2000 \
        --learning_rate 5e-4 \
        --loss_type mse \
        --seed 42 \
        --video \
        --video_interval 500
        
    echo "🎉 Training completed!"
else
    echo "❌ Teacher model validation failed. Please check the paths."
    exit 1
fi
