import base64
from PIL import Image, ImageDraw, ImageOps
from io import BytesIO
from openai import OpenAI
import ast,re,json

class_list=['步骤正确','图像认知错误','题意理解错误','缺乏相关知识','知识应用错误','逻辑过程错误','幻觉错误','运算处理错误','回答不完整错误']

class qwen25_vl_72b_instruct_judgemodel():
    def __init__(self, model_name, infer_mode, client_config,retry_times=10):
        self.client = OpenAI(
            api_key=client_config['api_key'],
            base_url=client_config['base_url'],
        )   
        self.infer_mode=infer_mode
        self.model_name=model_name
        self.retry_times=retry_times
    
    def directly_eval(self,response,mess) :    
        try:
            ground_truth=mess['format_answer']['ground_truth']
            pattern = r'<evaluation>(.*?)</evaluation>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                list_str = match.group(1).strip()
                list_str=re.sub(r'\\', r'\\\\', list_str)
                list_str = eval(list_str)
                assert type(list_str)==list , '结果格式错误'
                flag=1
                score=0
                assert len(list_str)==3 , '列表数目错误'
                for item in list_str[2]:
                    if int(item) not in [0,1]:
                        flag=0
                        break
                    else:
                        score+=int(item)
                score=score/len(list_str[2])
                assert score<=1 and score >=0, '分数不对'
                if flag==1:
                    return list_str,score
                else: 
                    return '',0
            else:
                return '',0
        except Exception as e:
            # print(f"解析错误：{e,response}")
            return '',0


    def step_by_step_eval(self,response) :
        try:
            index1 = response.find("[")
            index2 = response.rfind("]")
            response=response.replace('，',',')   
            if index1!=-1 and index2!=-1:
                try:
                    list_str = response[index1:index2+1]
                    list_str=re.sub(r'\\', r'\\\\', list_str)
                    list_str = eval(list_str)
                    assert type(list_str)==list , '结果格式错误'
                    if type(list_str[0][0])==list:
                        list_str=list_str[0]
                    flag='1'
                    acc=0
                    for item in list_str:
                        if type(item)!=list or len(item)!=3 or type(item[1])!=str  or item[1] not in class_list :
                            flag=item[1]
                            break    
                        acc+=1 if item[1]=='步骤正确' else 0
                    acc= acc/len(list_str)
                    if flag=='1':
                        return list_str,acc
                    else: 
                        # print(f"步骤错误{flag,response}")
                        return '',0
                except Exception as e:
                    # print(f"python错误{e,response}")
                    return '',0
            else:
                # print(f"未找到[]{response}")
                return '',0
        except Exception as e:
            # print(f"python错误{e,response}")
            return '',0
    def __call__(self, messes):
        def get_base64_image(path):
            image = Image.open(path).convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes)
            img_str = img_base64.decode('utf-8')
            return img_str

        prompt2gpt = messes["prompt2infer"].split('<image>')
        img_bin = messes["img"]
        prompt_img = [{
            'type':'image_url',
            'image_url':{
                'url':f"data:image/jpeg;base64,{img_bin}",
                'detail':'high'
            }}
        ]
        prompt_text_pre = [{'type':'text', 'text': prompt2gpt[0]}]
        prompt_text_end = [{'type':'text', 'text': prompt2gpt[1]}]
        # 图像在前，文本在后
        prompt = prompt_text_pre + prompt_img + prompt_text_end

        messages = [{'role': 'user', 'content': prompt}]

        score=0
        for i in range(self.retry_times):
            try:
                completion = self.client.chat.completions.create(
                    messages=messages, 
                    model=self.model_name,
                )
                response = completion.choices[0].message.content
                response,score=self.directly_eval(response,messes) if self.infer_mode=='directly' else self.step_by_step_eval(response)
                if response!='':
                    break
            except Exception as e:
                response = ""
                print(f"error:{e}")     
        return response,score


if __name__ == '__main__':
    print(qwen25_vl_72b_instruct_judgemodel('qwen25_vl_7b','directly',{
        'api_key':"EMPTY",
        'base_url':'http:xxx',
    }, )({'image':"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",
    'prompt2infer':'图片里有什么'}))