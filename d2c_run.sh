#!/usr/bin/env bash
# Stage 1c: the span-only arm. d2b changed --len_span AND --len_sigma together, so its
# result cannot be attributed to the span -- sigma is the exploration scale, and shrinking
# it starves the head of gradient signal (observed: p_sd settled at 1.0 under span 10 /
# sigma 3, versus 21 under span 25 / sigma 6).
#
# The justification for shrinking sigma does not survive arithmetic: sigma=6 contributes
# a realised p spread of 6 on its own, which the sigma sweep prices at about -0.001. That
# is negligible next to the -0.028 the span-25 head was paying, so sigma should have been
# held fixed.
set -e
RUN="conda run --no-capture-output -n annsrl python -u train_sift100k_ppo.py"
COMMON="--graph_type knn \
  --pretrained_path models/mlplink_SIFT100K_dot_best.pth \
  --max_steps 500 --commit_stride 1 --max_grad_norm 1.0 \
  --nodes_per_step 512 --dcs_budget 300 --ef 32 --k 10 \
  --logit_scale 20 --beta 0 --drop_mode policy \
  --accept off --actor_ctx --no_plot \
  --n_rand_cand 256 --max_periph_pct 95 --len_head --len_sigma 6"
mkdir -p logs
for S in 42 123 456; do
  CUDA_VISIBLE_DEVICES=0 $RUN $COMMON --seed $S --len_span 10 \
    --run_name d2c_span10sig6_s$S > logs/d2c_span10sig6_s$S.log 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=1 $RUN $COMMON --seed $S --len_span 15 \
    --run_name d2c_span15sig6_s$S > logs/d2c_span15sig6_s$S.log 2>&1 &
  P2=$!
  wait $P1 $P2
  echo "[D2c] seed $S done"
done
echo "[D2c] all done"
