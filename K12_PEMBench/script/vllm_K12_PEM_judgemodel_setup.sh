vllm \
serve  \
/global_data/mllm/models/Qwen2.5-VL-7B-Instruct   \
--task generate                              \
--served-model-name qwen25_vl_7b_instruct    \
--port 80 --trust-remote-code                \
--max_num_seqs 60                            \
--limit-mm-per-prompt image=8                \