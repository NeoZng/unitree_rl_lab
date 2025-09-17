@echo off
echo 🚀 Starting Multi-Teacher Distillation Training for Unitree G1...

REM 教师模型路径（请根据实际情况修改）
set TEACHER_PYRAMID=  C:\Users\BrainCo\Desktop\unitree_rl_lab\logs\rsl_rl\unitree_g1_29dof_velocity\2025-09-15_11-59-13\model_3600.pt
set TEACHER_SLOPE=    C:\Users\BrainCo\Desktop\unitree_rl_lab\logs\rsl_rl\unitree_g1_29dof_velocity\2025-09-15_11-59-13\model_3500.pt
set TEACHER_STAIRS=   C:\Users\BrainCo\Desktop\unitree_rl_lab\logs\rsl_rl\unitree_g1_29dof_velocity\2025-09-15_11-59-13\model_3400.pt
set TEACHER_GRID=     C:\Users\BrainCo\Desktop\unitree_rl_lab\logs\rsl_rl\unitree_g1_29dof_velocity\2025-09-15_11-59-13\model_3300.pt
set TEACHER_RANDOM=   C:\Users\BrainCo\Desktop\unitree_rl_lab\logs\rsl_rl\unitree_g1_29dof_velocity\2025-09-15_11-59-13\model_3200.pt
set TEACHER_WAVE=     C:\Users\BrainCo\Desktop\unitree_rl_lab\logs\rsl_rl\unitree_g1_29dof_velocity\2025-09-15_11-59-13\model_3100.pt

python .\scripts\rsl_rl\train_multi_teacher.py ^
    --task Isaac-Velocity-Distillation-G1-v0 ^
    --teacher_check ^
    --teacher_paths ^
        %TEACHER_PYRAMID% ^
        %TEACHER_SLOPE% ^
        %TEACHER_STAIRS% ^
        %TEACHER_GRID% ^
        %TEACHER_RANDOM% ^
        %TEACHER_WAVE% ^
    --num_envs 64 ^
    --max_iterations 2000 ^
    --learning_rate 5e-4 ^
    --loss_type mse ^
    --seed 42 
    
echo 🎉 Training completed!

