#!/usr/bin/env bash
# B1: random-init scorer — does the short-edge bias come from the pretrained weights?
# 3 seeds × 500 steps from kNN start, --scratch (no pretrained_path loaded)
# Compare final edge-length distribution and recall against ctrl_s{42,123,456} from Tier 1a.
set -e

COMMON="--graph_type knn \
  --scratch \
  --max_steps 500 --commit_stride 1 --max_grad_norm 1.0 \
  --nodes_per_step 512 --dcs_budget 300 --ef 32 --k 10 \
  --logit_scale 20 --beta 0 --drop_mode policy \
  --accept off --actor_ctx --no_plot"

echo "[B1] Launching 3 random-init runs (seeds 42, 123, 456) on GPUs 0/1 ..."

RUN="conda run --no-capture-output -n annsrl python -u train_sift100k_ppo.py"

CUDA_VISIBLE_DEVICES=0 $RUN $COMMON \
  --run_name b1_scratch_s42 --seed 42 > logs/b1_s42.log 2>&1 &
PID42=$!

CUDA_VISIBLE_DEVICES=1 $RUN $COMMON \
  --run_name b1_scratch_s123 --seed 123 > logs/b1_s123.log 2>&1 &
PID123=$!

echo "  s42  PID $PID42  → logs/b1_s42.log"
echo "  s123 PID $PID123 → logs/b1_s123.log"
echo "Waiting for both to finish before launching s456 ..."
wait $PID42 $PID123

CUDA_VISIBLE_DEVICES=0 $RUN $COMMON \
  --run_name b1_scratch_s456 --seed 456 > logs/b1_s456.log 2>&1 &
PID456=$!
echo "  s456 PID $PID456 → logs/b1_s456.log"
wait $PID456

echo "[B1] All done."
