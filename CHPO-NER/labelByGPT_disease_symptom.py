import openai
import threading
import time 
import json
from tqdm import tqdm
from func_timeout import func_set_timeout
import func_timeout


raw_text_path="hospital_data_json"###transfer EHR data to json file
save_path = 'GPT-NER-API/label_hospital_disease_symptom.json'
###加入自己的gpt账号api_key

openai.api_key = ''
repeat = 5      # 每个句子重复次数
save_every = 57 # 线程数目, 可视情况增大
sleep_time = 1  # 防止调用api超速

# 读取json文件
with open(raw_text_path, 'r') as of:
    content = of.read()
    lines = json.loads(content)
    pairs = []
    for line in lines:
        pairs.append((lines[line]['text'],lines[line]['mention'],lines[line]['span_gold']))
    of.close()

try:
    with open(save_path, 'r') as of:
        label_data = json.load(of)
except: label_data = {}


# 调用api 设置20秒超时

@func_set_timeout(20)
def gptapi(sent):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "请尽可能提取所有疾病和症状实体，同时忽略其他类型的实体。每个实体一行，以'-'开头，无需指出实体类型"},
            #{"role": "system", "content": "请为我识别 HPO（人类临床表型本体）实体。每个实体一行，以'-'开头。"},
            # {"role": "system",
            #  "content": "请尽可能提取所有 HPO（人类临床表型本体）实体，同时忽略其他类型的实体。每个实体一行，以'-'开头。"},
            # {"role": "system",
            #  "content": "请尽可能提取所有 HPO（人类临床表型本体）实体，同时忽略其他类型的实体。还请指出所提取实体的相应 HPO ID。每个实体一行，以'-'开头，并用';'分隔相关的 HPO ID。"},
            # {"role": "system",
            # "content": "请尽可能提取所有生物医学实体，同时忽略其他类型的实体。答案应该看起来像'entity(实体类型)'，每个实体一行以'-'开头。如果没有生物医学实体，则直接返回'-'。"},
            {"role": "user", "content": sent}
        ]
    )
    resText = response.choices[0].message.content
    return resText


label_result = [""] * save_every

# 多线程部分
class ChatThread(threading.Thread):
    def __init__(self, query, index):
        threading.Thread.__init__(self)
        self.query = query
        self.index = index

    def run(self):
        x = ""
        key = 1
        while len(x) == 0 & key:
            try:
                x = gptapi(self.query)
            except func_timeout.exceptions.FunctionTimedOut: 
                print("正在重试...")
                continue
        if key:
            label_result[self.index % save_every] = x
            key = 0
        return 1

idx = 0
sents = []
threads = []

for (sent,men,gold) in tqdm(pairs):
    if str(idx) in label_data:
        idx += 1
        continue

    label_data[str(idx)] = {
        'text': sent,
        'metion': men,
        'response': [],
        'span_gold': gold
    }

    sents.append(sent)

    if idx % save_every == save_every - 1:
        for j in range(repeat):
            for i in range(sents.__len__()):
                index = idx - sents.__len__() + 1 + i
                thread = ChatThread(sents[i], index)
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()

            for i in range(sents.__len__()):
                index = idx - sents.__len__() + 1 + i
                label_data[str(index)]['response'].append(label_result[index % save_every])
            
            label_result = [""] * save_every
            time.sleep(sleep_time)
        
        sents = []
        with open(save_path, 'w') as of:
            json.dump(label_data, of,ensure_ascii=False)
    idx += 1
        