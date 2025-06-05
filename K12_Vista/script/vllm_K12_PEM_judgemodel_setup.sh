vllm \
serve  \
/global_data/mllm/models/K12_PEM  \
--task generate                              \
--served-model-name K12_PEM    \
--port 80 --trust-remote-code                \
--max_num_seqs 60                            \
--limit-mm-per-prompt image=8                \