import json, os, glob, random, sys, argparse
import importlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from models import *
from model_dict import model_dict
from prompt import prompt_dict
import multiprocessing
lock = multiprocessing.Lock()

class K12_PEMBench():
    def __init__(self, infer_model, infer_prompt, in_path, out_dir, para_num):
        self.infer_model = infer_model
        self.infer_prompt=infer_prompt
        self.out_path = os.path.join(out_dir, self.infer_model,f"{infer_model}_result.jsonl")
        if not os.path.exists(self.out_path):
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.out_file=open(self.out_path,'a')
        self.metrix_json_out_path = os.path.join(out_dir, self.infer_model,f"{infer_model}_metrix.json")
        self.in_path = in_path
        self.para_num = para_num
        self.model = self._get_model(self.infer_model)
    
    def _get_model(self, model_name):
        try:
            module = importlib.import_module(f"models.vllminfer")
            model_class = getattr(module, 'vllminfer')
            print(f"module:{module}, model_class:{model_class}")
            return model_class( model_name, model_dict[model_name])
        except Exception as e:
            print(f'get_model_error:{e}')
    
    def _load_examples(self, in_path,out_path):
        try:
            done_items_ids=set()
            if not os.path.exists(out_path):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
            for line in open(out_path,"r"):
                try: 
                    done_items_ids.add(json.loads(line)['hash_id'])
                except Exception as e:
                    print(f'load_examples_error:{e}')

            data=[]
            for in_line in list(map(json.loads, open(in_path,"r"))):
                if in_line['hash_id'] not in done_items_ids:
                    data.append(in_line)
            self.example_num = len(data)
            return data
        except Exception as e :
            raise ValueError(f"Dataset eroor, please check data or in_path",e)

    def _build_prompt(self, tasks):
        for task in tasks:
            task['prompt2infer']=prompt_dict['step_by_step_'+self.infer_prompt].format(question=task['question'],solution=task['format_solution'],answer=task['ground_truth'],student_answer=task['student_infer'])
        return tasks
    def _infer_one(self, task):
        response = self.model(task)
        task['infer_result']={
            self.infer_model+'_response':response,
            "infer_model":self.infer_model
        }
        return task
    
    def _infer_parallel(self, tasks, para_num):
        cnt=0
        with ThreadPoolExecutor(para_num) as executor:
            for entry in tqdm(executor.map(self._infer_one, tasks), total=len(tasks),  \
                        desc=f'{self.infer_model} inference:'):
                if entry['infer_result'][self.infer_model+'_response']=='':
                    cnt+=1
                else: 
                    with lock:
                        self.out_file.write(json.dumps(entry,ensure_ascii=False)+"\n")  
        print(f'error_result_nums:{cnt}')
    
    def evaluate(self,input_path, metrix_json_out_path):
        data_in = list(map(json.loads, open(input_path,"r")))
        step_taxonomy={}
        for step_calss in ['all','步骤正确','图像认知错误','题意理解错误','缺乏相关知识','知识应用错误','逻辑过程错误','幻觉错误','运算处理错误','回答不完整错误']:
            step_taxonomy[step_calss]={
                'correct':0,
                'count':0,
                }
        results=[]
        for line in data_in:
            for step_predict,step_label in zip(line['infer_result'][self.infer_model+'_response'],line['step_labels']):
                #breakpoint()
                step_taxonomy[step_label[1]]['correct']+=1 if step_predict[1]==step_label[1] else 0
                step_taxonomy[step_label[1]]['count']+=1
                step_taxonomy['all']['correct']+=1 if step_predict[1]==step_label[1] else 0
                step_taxonomy['all']['count']+=1
        for step_calss in ['all','步骤正确','图像认知错误','题意理解错误','缺乏相关知识','知识应用错误','逻辑过程错误','幻觉错误','运算处理错误','回答不完整错误']:
            step_taxonomy[step_calss]=round(step_taxonomy[step_calss]['correct']/step_taxonomy[step_calss]['count'],4) if  step_taxonomy[step_calss]['count']!=0 else 0
        metrix_json_out_file=open(metrix_json_out_path,'w')
        json.dump(step_taxonomy, metrix_json_out_file, ensure_ascii=False, indent=4)
        print(step_taxonomy)
       


    def __call__(self):
        datas = self._load_examples(self.in_path, self.out_path)
        input_datas = self._build_prompt(datas)
        self._infer_parallel(input_datas, self.para_num)
        self.evaluate(self.out_path, self.metrix_json_out_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", type=str, default="qwen25_vl_72b_instruct")
    parser.add_argument("--infer_prompt", type=str, default="directly")
    parser.add_argument("--in_path", type=str, default="../data/K12_Vista.jsonl")
    parser.add_argument("--out_dir", type=str, default="../output/respone")
    parser.add_argument("--max_threads", type=int, default=10)
    args = parser.parse_args()
    K12_PEMBench_eval = K12_PEMBench(args.infer_model, args.infer_prompt, args.in_path, args.out_dir, args.max_threads)
    K12_PEMBench_eval()