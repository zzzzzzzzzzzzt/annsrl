python pretrain.py --dataset DEEP100K --graph_type nsw \
--vertices_path data/DEEP100K/deep_base.fvecs \
--edges_path data/DEEP100K/deep_hnsw_M12_efC300.ivecs \
--rand_split --method nodeformer --lr 0.005 \
--weight_decay 0.05 --dropout 0.3 --num_layers 3 \
--hidden_channels 256 --num_heads 5 --rb_order 0 \
--rb_trans sigmoid --lamda 0 --M 30 --K 10 --use_bn \
--use_residual --use_gumbel --runs 5 --epochs 300 --device 0 