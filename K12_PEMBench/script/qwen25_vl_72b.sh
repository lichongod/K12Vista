in_path='data/K12_PEMbench.jsonl'
out_dir='result'
max_threads=60
infer_model='qwen25_vl_72b_instruct'
infer_prompt='infer'

python code/infer.py    \
--infer_model=${infer_model}    \
--infer_prompt=${infer_prompt}     \
--in_path=${in_path}     \
--out_dir=${out_dir}   \
--max_threads=${max_threads}

