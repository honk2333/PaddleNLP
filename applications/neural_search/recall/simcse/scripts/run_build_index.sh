# gpu
python -u -m paddle.distributed.launch --gpus "0,1,2" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result3.txt" \
        --params_path "checkpoints/model_6000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 128 \
        --output_emb_size 128\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "../dev.csv" \
        --corpus_file "../train_unsupervised.csv"

# cpu
# python  recall.py \
#         --device cpu \
#         --recall_result_dir "recall_result_dir" \
#         --recall_result_file "recall_result.txt" \
#         --params_path "checkpoints/model_20000/model_state.pdparams" \
#         --hnsw_m 100 \
#         --hnsw_ef 100 \
#         --batch_size 64 \
#         --output_emb_size 256\
#         --max_seq_length 60 \
#         --recall_num 50 \
#         --similar_text_pair "recall/dev.csv" \
#         --corpus_file "recall/corpus.csv" 