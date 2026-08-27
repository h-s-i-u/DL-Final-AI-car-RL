# Map3 ablation sweep — 4 conditions, seed 42, 500k steps each
$ErrorActionPreference = "Continue"
$start = Get-Date

Write-Host "=== Run 1/4: baseline ===" -ForegroundColor Cyan
python train.py --exp baseline    --map assets/maps/map3.png --timesteps 500000 --seed 42 --n-envs 16 --vec subproc

Write-Host "=== Run 2/4: reward_only ===" -ForegroundColor Cyan
python train.py --exp reward_only --map assets/maps/map3.png --timesteps 500000 --seed 42 --n-envs 16 --vec subproc

Write-Host "=== Run 3/4: obs_only ===" -ForegroundColor Cyan
python train.py --exp obs_only    --map assets/maps/map3.png --timesteps 500000 --seed 42 --n-envs 16 --vec subproc

Write-Host "=== Run 4/4: full ===" -ForegroundColor Cyan
python train.py --exp full        --map assets/maps/map3.png --timesteps 500000 --seed 42 --n-envs 16 --vec subproc

$elapsed = (Get-Date) - $start
Write-Host "=== ALL DONE in $($elapsed.TotalMinutes) min ===" -ForegroundColor Green