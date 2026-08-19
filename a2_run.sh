#!/usr/bin/env bash
# A2: can the policy add anything ON TOP of the free random-rewiring gain?
#
# s_0 = kNN + 5% random rewiring (recall 0.524 at budget=300), written by
# make_rewired_s0.py. If training ends ABOVE that, the policy contributes
# something random rewiring does not. If it flattens or decays, it does not --
# and "+0.04 over kNN s_0" was only ever recovering part of what one line of
# numpy supplies for free.
#
# Same config as the Tier 1a control arm, so the only change is s_0.
set -u
PYTHON=/mnt/HDD0/home/zjw25/anaconda3/envs/annsrl/bin/python
cd /mnt/HDD0/home/zjw25/annsrl
mkdir -p logs

COMMON="--graph_type knn \
  --init_graph_from runs/rewired_s0/dynamic_edges.0.pth \
  --pretrained_path models/mlplink_SIFT100K_dot_best.pth \
  --max_steps 500 --commit_stride 1 --max_grad_norm 1.0 \
  --nodes_per_step 512 --dcs_budget 300 --ef 32 --k 10 \
  --logit_scale 20 --beta 0 --drop_mode policy \
  --accept off --actor_ctx --no_plot"

i=0
for SEED in 42 123 456; do
  GPU=$(( i % 2 )); i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u train_sift100k_ppo.py $COMMON \
    --seed "$SEED" --init_seed "$SEED" \
    --run_name "a2_rewired_s${SEED}" \
    > "logs/a2_rewired_s${SEED}.log" 2>&1 &
  echo "launched a2_rewired_s${SEED} on GPU $GPU"
done
wait
echo "=== A2 ALL RUNS COMPLETE ==="
