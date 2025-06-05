vllm \
serve  \
/global_data/mllm/models/Qwen2.5-VL-72B-Instruct   \
--task generate                              \
--served-model-name qwen25_vl_72b_instruct_judgemodel   \
--port 80 --trust-remote-code                \
--max_num_seqs 60                            \
--limit-mm-per-prompt image=8                \
--tensor-parallel-size 4