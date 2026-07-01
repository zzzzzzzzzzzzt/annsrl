#!/usr/bin/env bash
set -euo pipefail

ACTIVATIONS=("Sigmoid" "Tanh")
FACTORS=("1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0")

RUNS=5
EPOCHS=300
EVAL_STEP=5
DEVICE=0
TOPN_BATCH_SIZE=256

RESULT_ROOT="results/Topology_factor&Activation_Experiment"
LOG_DIR="${RESULT_ROOT}/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

for activation in "${ACTIVATIONS[@]}"; do
  for factor in "${FACTORS[@]}"; do
    log_path="${LOG_DIR}/${activation}_factor${factor}.log"
    echo "[RUN] activation=${activation}, topology_factor=${factor}"

    python3 pretrain.py --dataset DEEP10K --graph_type nsw \
      --vertices_path data/DEEP100K/deep10k/deep_base_random10000_seed42.fvecs \
      --edges_path data/DEEP100K/deep10k/deep_hnsw_M12_efC300_random10000_seed42.ivecs \
      --rand_split --method nodeformer --lr 0.005 \
      --weight_decay 0.05 --dropout 0.3 --num_layers 2 \
      --hidden_channels 256 --num_heads 3 --rb_order 0 \
      --rb_trans sigmoid --lamda 0 --M 30 --K 10 --use_bn \
      --use_residual --use_gumbel --runs "${RUNS}" --epochs "${EPOCHS}" \
      --device "${DEVICE}" --eval_step "${EVAL_STEP}" \
      --negative_hop 2 --tau 0.5 --topology_activation "${activation}" \
      --topology_factor "${factor}" --loss_function degree_log \
      --topn_batch_size "${TOPN_BATCH_SIZE}" 2>&1 | tee "${log_path}"
  done
done

python3 plot_topology_stable.py \
  --root "${RESULT_ROOT}" \
  --metric test_mass \
  --select_metric valid_mass \
  --window 8 \
  --out "${RESULT_ROOT}/topology_activation_factor_mass_window40epoch.svg" \
  --summary "${RESULT_ROOT}/topology_activation_factor_mass_window40epoch.csv"

python3 plot_topology_stable.py \
  --root "${RESULT_ROOT}" \
  --metric test_topn_ratio \
  --select_metric valid_topn_ratio \
  --window 8 \
  --out "${RESULT_ROOT}/topology_activation_factor_topn_window40epoch.svg" \
  --summary "${RESULT_ROOT}/topology_activation_factor_topn_window40epoch.csv"

echo "[DONE] logs saved to ${LOG_DIR}"
