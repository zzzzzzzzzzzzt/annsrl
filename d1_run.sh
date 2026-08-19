#!/usr/bin/env bash
# D1: the rule sets the edge length, the policy chooses within the band.
#
# Motivated by two measured facts, not by hope:
#   C2  the length optimum is INTERIOR at percentile ~65, and the learning-free bar
#       (kNN + 5% rewire to pct 65) is 0.557 +- 0.008 -- what any arm must beat.
#   C1  inside a fixed length, the policy's target choice is worth +0.010 to +0.014,
#       confirmed against a same-length reshuffle and a placebo. That is the ONLY
#       thing it has ever been shown to do, so it is the only thing it is asked to do.
#
# long_frac is deliberately OFF: it takes the longest fraction of the 2-hop pool, which
# tops out at per-node percentile 8.95 on kNN, so it cannot express a band at 65. The
# random candidates are what make the band reachable (verified: 0% in band without them).
#
# Arm 2 adds the anti-peripherality filter. Kept separate so its marginal contribution
# is measurable rather than bundled -- the hubs were centroid-pct 98.8-99.8 outliers.
set -e
RUN="conda run --no-capture-output -n annsrl python -u train_sift100k_ppo.py"
COMMON="--graph_type knn \
  --pretrained_path models/mlplink_SIFT100K_dot_best.pth \
  --max_steps 500 --commit_stride 1 --max_grad_norm 1.0 \
  --nodes_per_step 512 --dcs_budget 300 --ef 32 --k 10 \
  --logit_scale 20 --beta 0 --drop_mode policy \
  --accept off --actor_ctx --no_plot \
  --cand_band 55 75 --n_rand_cand 256"
mkdir -p logs

for S in 42 123 456; do
  CUDA_VISIBLE_DEVICES=0 $RUN $COMMON --seed $S \
    --run_name d1_band_s$S > logs/d1_band_s$S.log 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=1 $RUN $COMMON --seed $S --max_periph_pct 95 \
    --run_name d1_periph_s$S > logs/d1_periph_s$S.log 2>&1 &
  P2=$!
  wait $P1 $P2
  echo "[D1] seed $S done"
done
echo "[D1] all done. Score with:"
echo "  python eval_harness.py --glob 'runs/d1_*/dynamic_edges.500.pth'"
