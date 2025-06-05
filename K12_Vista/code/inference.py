#!/usr/bin/env python
#coding:utf-8

import json, os, glob, random, sys, argparse
import importlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from models import *
from model_dict import model_dict
from prompt import infer_prompt
import multiprocessing
lock = multiprocessing.Lock()

class K12_Vista():
    def __init__(self, infer_model, infer_mode, in_path, out_dir, para_num):
        self.infer_model = infer_model
        self.infer_mode=infer_mode
        self.out_path = os.path.join(out_dir, infer_mode+'_infer',f"{infer_model}_infer.jsonl")
        if not os.path.exists(self.out_path):
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.out_file=open(self.out_path,'a')
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
            task['prompt2infer']=infer_prompt[self.infer_mode+'_infer_prompt'][task['type']].format(question=task['question'])
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


    def __call__(self):
        datas = self._load_examples(self.in_path, self.out_path)
        input_datas = self._build_prompt(datas)
        self._infer_parallel(input_datas, self.para_num)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", type=str, default="qwen25_vl_72b_instruct")
    parser.add_argument("--infer_mode", type=str, default="directly")
    parser.add_argument("--in_path", type=str, default="../data/K12_Vista.jsonl")
    parser.add_argument("--out_dir", type=str, default="../output/respone")
    parser.add_argument("--max_threads", type=int, default=10)
    args = parser.parse_args()
    K12_Vista_infer = K12_Vista(args.infer_model, args.infer_mode, args.in_path, args.out_dir, args.max_threads)
    K12_Vista_infer()