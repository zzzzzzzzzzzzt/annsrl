#!/usr/bin/env bash
# Tier 1a: does the in-degree brake stop hub growth and improve recall?
#
# Paired A/B on the kNN start, which is where the runaway hubs were measured
# (max in-degree 8039 vs NSW's 71). accept=off because that is the regime the
# explosion was observed in -- accept=node masks it by rolling back the harmful
# edits, and is harmful at deploy time anyway.
#
#   arm A (ctrl):  --actor_ctx                            (current best config)
#   arm B (ideg):  --actor_ctx --indeg_ctx --indeg_noop
#
# Both arms share --seed, so the node sample and query batch of every step are
# identical and the pair differs only in the two zero-init heads. The heads are
# RNG-neutral (nn.Parameter(zeros), not nn.Linear), so step 0 is bit-identical.
set -u

PYTHON=/mnt/HDD0/home/zjw25/anaconda3/envs/annsrl/bin/python
cd /mnt/HDD0/home/zjw25/annsrl
mkdir -p logs

COMMON="--graph_type knn \
  --pretrained_path models/mlplink_SIFT100K_dot_best.pth \
  --max_steps 500 --commit_stride 1 --max_grad_norm 1.0 \
  --nodes_per_step 512 --dcs_budget 300 --ef 32 --k 10 \
  --logit_scale 20 --beta 0 --drop_mode policy \
  --accept off --actor_ctx --no_plot"

SEEDS="42 123 456 789"
i=0
for SEED in $SEEDS; do
  # GPUs 0 and 1 are free (2 is occupied by another job); alternate between them.
  GPU=$(( i % 2 ))
  i=$(( i + 1 ))

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u train_sift100k_ppo.py $COMMON \
    --seed "$SEED" --init_seed "$SEED" \
    --run_name "tier1a_ctrl_s${SEED}" \
    > "logs/tier1a_ctrl_s${SEED}.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u train_sift100k_ppo.py $COMMON \
    --seed "$SEED" --init_seed "$SEED" \
    --indeg_ctx --indeg_noop \
    --run_name "tier1a_ideg_s${SEED}" \
    > "logs/tier1a_ideg_s${SEED}.log" 2>&1 &

  echo "launched seed $SEED (both arms) on GPU $GPU"
done

echo "8 runs launched; waiting..."
wait
echo "=== TIER1A ALL RUNS COMPLETE ==="
