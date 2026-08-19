#!/usr/bin/env bash
# B2 stage 2: seeds for the two fracs worth replicating.
#
# Stage 1 (seed 42) gave recall 0.3964 / 0.3045 / 0.4801 for frac 0.25 / 0.10 / 0.01.
# That ordering is NON-MONOTONE -- 0.10 came out below 0.25 despite being the more
# aggressive filter -- so at n=1 it cannot be told apart from seed noise. Replicating
# 0.01 (the best) and 0.10 (the anomaly) decides whether the frac->recall curve is
# real or whether stage 1 just drew a lucky and an unlucky seed.
set -e

RUN="conda run --no-capture-output -n annsrl python -u train_sift100k_ppo.py"
COMMON="--graph_type knn \
  --pretrained_path models/mlplink_SIFT100K_dot_best.pth \
  --max_steps 500 --commit_stride 1 --max_grad_norm 1.0 \
  --nodes_per_step 512 --dcs_budget 300 --ef 32 --k 10 \
  --logit_scale 20 --beta 0 --drop_mode policy \
  --accept off --actor_ctx --no_plot"

mkdir -p logs
for SEED in 123 456; do
  CUDA_VISIBLE_DEVICES=0 $RUN $COMMON --long_frac 0.01 --seed $SEED \
    --run_name b2_frac001_s$SEED > logs/b2_frac001_s$SEED.log 2>&1 &
  A=$!
  CUDA_VISIBLE_DEVICES=1 $RUN $COMMON --long_frac 0.10 --seed $SEED \
    --run_name b2_frac010_s$SEED > logs/b2_frac010_s$SEED.log 2>&1 &
  B=$!
  echo "  seed $SEED: frac001 PID $A ; frac010 PID $B"
  wait $A $B
done
echo "[B2] stage 2 done."
