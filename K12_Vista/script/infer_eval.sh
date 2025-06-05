
infer_model='qwen25_vl_7b_instruct'
in_path='data/k12_Vista.jsonl'
max_threads=50

# infer_mode='directly'
# eval_model="qwen25_vl_72b_instruct_judgemodel"

# python code/inference.py  \
#     --infer_model=${infer_model}    \
#     --infer_mode=${infer_mode}     \
#     --in_path=${in_path}     \
#     --out_dir='output/response'    \
#     --max_threads=${max_threads}

# python code/evalaute.py  \
#     --infer_model=${infer_model}    \
#     --eval_model=${eval_model}    \
#     --infer_mode=${infer_mode}     \
#     --in_dir="output/response"   \
#     --out_dir="output/judge"    \
#     --max_threads=${max_threads} 

#############step-by-step eval

infer_mode='step_by_step'
eval_model="K12_PEM_judgemodel"

python code/inference.py  \
    --infer_model=${infer_model}    \
    --infer_mode=${infer_mode}     \
    --in_path=${in_path}     \
    --out_dir='output/response'    \
    --max_threads=${max_threads} 

# python code/evalaute.py  \
#     --infer_model=${infer_model}    \
#     --eval_model=${eval_model}    \
#     --infer_mode=${infer_mode}     \
#     --in_dir="output/response"   \
#     --out_dir="output/judge"    \
#     --max_threads=${max_threads} 
