python3 build_sample_graph.py \
--vertices_path data/DEEP100K/deep_base.fvecs \
--output_dir data/DEEP100K/deep10k \
--sample_size 10000 \
--seed 42 \
--sample_method random \
--graph_method hnsw \
--hnsw_m 12 \
--hnsw_ef_construction 300 \
--hnsw_ef 300
