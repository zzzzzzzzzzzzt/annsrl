#!/usr/bin/env bash
# Stage 1 / D2: learn the length percentile p_u per node.
#
# Stage 0 measured the learning-free version of this and found NULL -- but only for
# LINEAR maps of three closed-form features (knn_radius / centroid / lid). This is the
# function-class-free version: the head sees raw z_u plus its neighbourhood mean, so it
# can express any relation those three scalars cannot.
#
# Zero-init output layer => the starting policy IS p=65 for every node, i.e. exactly
# C2's measured optimum. Anything the head does is a strict improvement test over the
# rule, not a from-scratch gamble.
#
# ARM 2 IS THE POINT. --len_fixed 65 uses the SAME per-node rank band with nothing
# learned, so head-vs-control isolates the head. Comparing against D1 instead would
# confound it with the band definition changing from global-CDF to per-node rank (a
# discrepancy found while designing this: C2 optimised a per-node rank, D1 shipped a
# global CDF, and they coincide only for a perfectly homogeneous dataset).
#
# Judged by stage1_eval.py: beat p=65, beat its OWN shuffled p_u, and not merely
# re-derive the Stage 0 features.
set -e
RUN="conda run --no-capture-output -n annsrl python -u train_sift100k_ppo.py"
COMMON="--graph_type knn \
  --pretrained_path models/mlplink_SIFT100K_dot_best.pth \
  --max_steps 500 --commit_stride 1 --max_grad_norm 1.0 \
  --nodes_per_step 512 --dcs_budget 300 --ef 32 --k 10 \
  --logit_scale 20 --beta 0 --drop_mode policy \
  --accept off --actor_ctx --no_plot \
  --n_rand_cand 256 --max_periph_pct 95"
mkdir -p logs

for S in 42 123 456; do
  CUDA_VISIBLE_DEVICES=0 $RUN $COMMON --seed $S --len_head \
    --run_name d2_head_s$S > logs/d2_head_s$S.log 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=1 $RUN $COMMON --seed $S --len_fixed 65 \
    --run_name d2_fixed_s$S > logs/d2_fixed_s$S.log 2>&1 &
  P2=$!
  wait $P1 $P2
  echo "[D2] seed $S done"
done
echo "[D2] all done"
