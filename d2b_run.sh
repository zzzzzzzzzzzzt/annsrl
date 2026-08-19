#!/usr/bin/env bash
# Stage 1b: the head's information is real, its spread is the problem.
#
# Pooled over 9 cells, the learned p_u beats its OWN shuffle by +0.0321 +- 0.0093
# (t=3.5) -- the assignment carries real information. But at span=25 the head drove
# p_u to sd~19-21 with tanh saturated at both ends (p5 40, p95 89), and p~90 is the
# dead-end region C2 measured at 0.5126. That spread costs -0.0282 +- 0.0086 against
# the delta-at-65 rule, which eats almost exactly what the assignment earned: net
# +0.0039 +- 0.0026, unresolved.
#
# So constrain the span. At span=10 the head can only differentiate inside [55, 75],
# where the sigma sweep puts the tax near -0.003 instead of -0.028. If even half the
# +0.032 survives, the net is clearly positive.
#
# This is a hyperparameter of the ACTION SPACE, not of the network: it bounds what the
# head may do, and the zero-init means every span still starts at exactly p=65.
set -e
RUN="conda run --no-capture-output -n annsrl python -u train_sift100k_ppo.py"
COMMON="--graph_type knn \
  --pretrained_path models/mlplink_SIFT100K_dot_best.pth \
  --max_steps 500 --commit_stride 1 --max_grad_norm 1.0 \
  --nodes_per_step 512 --dcs_budget 300 --ef 32 --k 10 \
  --logit_scale 20 --beta 0 --drop_mode policy \
  --accept off --actor_ctx --no_plot \
  --n_rand_cand 256 --max_periph_pct 95 --len_head"
mkdir -p logs

for S in 42 123 456; do
  CUDA_VISIBLE_DEVICES=0 $RUN $COMMON --seed $S --len_span 10 --len_sigma 3 \
    --run_name d2b_span10_s$S > logs/d2b_span10_s$S.log 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=1 $RUN $COMMON --seed $S --len_span 15 --len_sigma 4 \
    --run_name d2b_span15_s$S > logs/d2b_span15_s$S.log 2>&1 &
  P2=$!
  wait $P1 $P2
  echo "[D2b] seed $S done"
done
echo "[D2b] all done"
