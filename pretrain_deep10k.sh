python3 pretrain.py --dataset DEEP10K --graph_type nsw \
--vertices_path data/DEEP100K/deep10k/deep_base_random10000_seed42.fvecs \
--edges_path data/DEEP100K/deep10k/deep_hnsw_M12_efC300_random10000_seed42.ivecs \
--rand_split --method nodeformer --lr 0.005 \
--weight_decay 0.05 --dropout 0.3 --num_layers 2 \
--hidden_channels 128 --num_heads 2 --rb_order 0 \
--rb_trans sigmoid --lamda 0 --M 30 --K 10 --use_bn \
--use_residual --use_gumbel --runs 5 --epochs 300 --device 0
