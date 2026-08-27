# eval_all.ps1 — cross-map evaluation
$models = @(
    "logs/ppo_baseline_map_20260527_010212_s42/final_model.zip"
    # 加上等等 map3 跑完的 4 個 path
)
$maps = @(
    "assets/maps/map.png",
    "assets/maps/map3.png"
)

foreach ($m in $models) {
    foreach ($map in $maps) {
        Write-Host "=== Eval $m on $map ===" -ForegroundColor Cyan
        python eval.py $m --map $map --episodes 20
    }
}