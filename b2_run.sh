#!/usr/bin/env bash
# B2: force long edges by restricting the candidate menu.
#
# Stage 1 (this script) sweeps long_frac at ONE seed. The filter shifts the pool's
# mean length percentile 2.54 -> 4.65 / 5.85 / 8.38 for frac 0.25 / 0.10 / 0.01, so
# these three bracket "mild nudge" to "almost forced onto the longest edge".
#
# Only if some frac beats the count-matched random control does stage 2 add seeds.
# Running 3 seeds x 3 fracs up front would spend 9 runs to learn what 3 can rule out.
set -e

RUN="conda run --no-capture-output -n annsrl python -u train_sift100k_ppo.py"
COMMON="--graph_type knn \
  --pretrained_path models/mlplink_SIFT100K_dot_best.pth \
  --max_steps 500 --commit_stride 1 --max_grad_norm 1.0 \
  --nodes_per_step 512 --dcs_budget 300 --ef 32 --k 10 \
  --logit_scale 20 --beta 0 --drop_mode policy \
  --accept off --actor_ctx --no_plot --seed 42"

mkdir -p logs

echo "[B2] stage 1: long_frac sweep at seed 42"
CUDA_VISIBLE_DEVICES=0 $RUN $COMMON --long_frac 0.25 \
  --run_name b2_frac025_s42 > logs/b2_frac025_s42.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 $RUN $COMMON --long_frac 0.10 \
  --run_name b2_frac010_s42 > logs/b2_frac010_s42.log 2>&1 &
P2=$!
echo "  frac 0.25 PID $P1 ; frac 0.10 PID $P2"
wait $P1 $P2

CUDA_VISIBLE_DEVICES=0 $RUN $COMMON --long_frac 0.01 \
  --run_name b2_frac001_s42 > logs/b2_frac001_s42.log 2>&1 &
P3=$!
echo "  frac 0.01 PID $P3"
wait $P3

echo "[B2] stage 1 done. Score with:"
echo "  python eval_harness.py --glob 'runs/b2_frac*_s42/dynamic_edges.500.pth'"

# Stage 2 (invoked separately): seeds for the fracs stage 1 singled out.
