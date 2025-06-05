#!/usr/bin/env python
#coding:utf-8

import os, random, json
from tqdm import tqdm
import pandas as pd
import importlib
from concurrent.futures import ThreadPoolExecutor
import argparse
random.seed(9876)
from models import *
from model_dict import model_dict
from prompt import eval_prompt
import multiprocessing
lock = multiprocessing.Lock()
 

class Evaluation():
    def __init__(self, infer_model, infer_mode, in_dir, out_dir, para_num, eval_model):
        self.infer_model = infer_model
        self.infer_mode = infer_mode
        self.eval_model_name = eval_model
        self.in_path = os.path.join(in_dir, infer_mode+'_infer',f"{infer_model}_infer.jsonl")
        self.out_path = os.path.join(out_dir, infer_mode+'_eval',f"{infer_model}_eval.jsonl")
        if not os.path.exists(self.out_path):
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.out_file=open(self.out_path,'a')
        self.excel_out_path = os.path.join(out_dir, infer_mode+'_eval_metric',f"{infer_model}_eval.xlsx")
        if not os.path.exists(self.excel_out_path):
            os.makedirs(os.path.dirname(self.excel_out_path), exist_ok=True)
        self.para_num = para_num
        self.eval_model = self._get_model(self.eval_model_name)
        print(f"infer_model:{self.infer_model}\t in_path:{self.in_path}\t out_path:{self.out_path}\t" 
              f"excel_out_path:{self.excel_out_path}\t para_num:{self.para_num}")

    def _get_model(self, model_name):
        try:
            module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(module, model_name)
            return model_class(model_name, self.infer_mode, model_dict[model_name])
        except (ImportError, AttributeError) as e:
            raise ValueError(f"{model_name} is not defined: {e}, Please ensure that module_name is equal to model_name and that both have been defined.")
        except Exception as e:
            print(f'error:{e}')

    def _load_examples(self, in_path,out_path):
        try:
            done_item_ids=set()
            for out_line in open(out_path,"r"):
                try:
                    done_item_ids.add(json.loads(out_line)['hash_id'])
                except Exception as e:
                    print(f'get_model_error:{e}')

            data=[]
            for in_line in open(in_path,"r"):
                try: 
                    line=json.loads(in_line)
                    if line['hash_id'] not in done_item_ids:
                        data.append(line)
                except Exception as e:
                    print(f'get_model_error:{e}')
            self.example_num = len(data)
            return data
        except Exception as e :
            raise ValueError(f"Dataset eroor, please check data or in_path",e)

    def _build_prompt(self, tasks):
        for task in tasks:
            task['prompt2infer']=eval_prompt[self.infer_mode+'_eval_prompt'][task['type']].format(question=task['question'],answer=task['format_answer']['ground_truth'],solution=task['format_answer']['format_solution'],student_answer=task['infer_result'][self.infer_model+'_response'])
        return tasks

    def _judge_one(self, task):    
        response,score = self.eval_model(task)
        task['judgement_result']={
            self.infer_model+'_judged_response':response,
            self.infer_model+'_judged_score':score,
            "judgement_model":self.eval_model_name,
            'infer_model':self.infer_model,
        }
        return task

    def _judged_parallel(self, tasks, para_num):
        cnt=0
        with ThreadPoolExecutor(para_num) as executor:
            for entry in tqdm(executor.map(self._judge_one, tasks), total=len(tasks), desc=f'eval {self.infer_model}'):
                if entry['judgement_result'][self.infer_model+'_judged_response']=='':
                    cnt+=1
                else:
                    with lock:
                        self.out_file.write(json.dumps(entry,ensure_ascii=False)+"\n")  
        
    def _save_result(self, result):
        question_subjects = ['math-g6','math-g9','math-g12','physics-g9','physics-g12','chemistry-g9','chemistry-g12','biology-g9','biology-g12','geography-g9','geography-g12',]
        question_types = ['填空题','问答题','选择题']

        datas=[]
        for out_line in open(self.out_path,"r"):
            try: 
                datas.append(json.loads(out_line))
            except Exception as e:
                print(f'eval_result_input_error:{e}')
        print(f'result_nums:{len(datas)}')


        #datas=list(map(json.loads, open(self.out_path,"r")))
        taxonomy_dict={}
        for question_type in question_types:
            taxonomy_dict[question_type]={}
            for question_subject in question_subjects:
                taxonomy_dict[question_type][question_subject]={
                    'score_all':0,
                    'count':0,
                }
        for line in datas:
            taxonomy_dict[line['type']][line['subject']]['score_all']+=line['judgement_result'][self.infer_model+'_judged_score']
            taxonomy_dict[line['type']][line['subject']]['count']+=1
        result_metric={'选择题':{},'填空题':{},'问答题':{},'所有题型':{}}
        for key in question_subjects+['g6','g9','g12']+['math','physics','chemistry','biology','geography','所有(sub*grade)']:
            result_metric['所有题型'][key]={'score_all':0,'count':0,}
        for question_type in question_types:
            for question_subject in question_subjects:
                result_metric[question_type][question_subject]=taxonomy_dict[question_type][question_subject]
                result_metric['所有题型'][question_subject]['score_all']+=taxonomy_dict[question_type][question_subject]['score_all']
                result_metric['所有题型'][question_subject]['count']+=taxonomy_dict[question_type][question_subject]['count']
            for question_grade in ['g6','g9','g12']:
                result_metric[question_type][question_grade]={'score_all':0,'count':0,}
            for question_subject in question_subjects:
                    result_metric[question_type][question_subject.split('-')[1]]['score_all']+=taxonomy_dict[question_type][question_subject]['score_all']
                    result_metric[question_type][question_subject.split('-')[1]]['count']+=taxonomy_dict[question_type][question_subject]['count']
            for question_grade in ['g6','g9','g12']:
                result_metric['所有题型'][question_grade]['score_all']+=result_metric[question_type][question_grade]['score_all']
                result_metric['所有题型'][question_grade]['count']+=result_metric[question_type][question_grade]['count']
                            
            for tax_subject in ['math','physics','chemistry','biology','geography','所有(sub*grade)']:
                result_metric[question_type][tax_subject]={'score_all':0,'count':0,}
            for question_subject in question_subjects:
                result_metric[question_type][question_subject.split('-')[0]]['score_all']+=taxonomy_dict[question_type][question_subject]['score_all']
                result_metric[question_type][question_subject.split('-')[0]]['count']+=taxonomy_dict[question_type][question_subject]['count']
                result_metric[question_type]['所有(sub*grade)']['score_all']+=taxonomy_dict[question_type][question_subject]['score_all']
                result_metric[question_type]['所有(sub*grade)']['count']+=taxonomy_dict[question_type][question_subject]['count']
            for tax_subject in ['math','physics','chemistry','biology','geography']:
                result_metric['所有题型'][tax_subject]['score_all']+=result_metric[question_type][tax_subject]['score_all']
                result_metric['所有题型'][tax_subject]['count']+=result_metric[question_type][tax_subject]['count']
            result_metric['所有题型']['所有(sub*grade)']['score_all']+=result_metric[question_type]['所有(sub*grade)']['score_all']
            result_metric['所有题型']['所有(sub*grade)']['count']+=result_metric[question_type]['所有(sub*grade)']['count']
        for item in result_metric.values():
            for key in  item.keys():
                item[key]=item[key]['score_all']/item[key]['count']

        print(result_metric)
        df = pd.DataFrame.from_dict(result_metric, orient='index')
        df = df.reset_index().rename(columns={'index': '类型'})
        df.to_excel(self.excel_out_path, index=False)
            

              
    def __call__(self):
        datas = self._load_examples(self.in_path, self.out_path)
        input_datas = self._build_prompt(datas)
        self._judged_parallel(input_datas, self.para_num)
        self._save_result()    
        # judge


  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", type=str, default="moonshot")
    parser.add_argument("--infer_mode", type=str, default="directly")
    parser.add_argument("--in_dir", type=str, default="data/6w.jsonl")
    parser.add_argument("--out_dir", type=str, default="output/response")
    parser.add_argument("--max_threads", type=int, default=10)
    parser.add_argument("--eval_model", type=str, default='gpt4o')
    args = parser.parse_args()

    Evaluation(args.infer_model, args.infer_mode, args.in_dir, args.out_dir, args.max_threads, args.eval_model)()