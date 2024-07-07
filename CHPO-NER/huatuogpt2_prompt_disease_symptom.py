import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import os
# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3950
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



pathmain=""


tokenizer = AutoTokenizer.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B", use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B")


path_patient=pathmain+"/hospital_data"###EHR data
path_result= "HuatuoGPT2-NER-API/label_HuatuoGPT2_hospital_disease_symptom.json"



path_result_list=[]
filelist_d=os.listdir(path_result)
for file in filelist_d:
    path_result_list.append(file)


filelist=os.listdir(path_patient)
Npine=0
for file in filelist:
    torch.cuda.empty_cache()

    file_name=file.split(".")[0]

    if file_name not in path_result_list:

        # sent="Clinicians should suspect NF2 in children presenting with meningioma, schwannoma, and skin features, such as neurofibromas/schwannomas, but fewer than 6 caf茅 au lait patches, who thus fall short of a diagnosis of neurofibromatosis type 1."
        # #
        # sent = ("【临床表现】受累肢体局部的疼痛并伴有肿块肿块的增大可以非常迅速，局部表皮发红并伴有温度升高骨髓炎的表现相似。"
        #         "早期的症状有时同病变部位有着较大的关系，肋骨受累的病例可以有胸腔积液，下颌骨受到侵犯时可以有局部区域的感觉麻木，脊柱浸润时也可以因脊髓受压而产生下肢的无力甚至瘫痪X线摄片和CT扫描对于明确肿瘤的大小和性质有很大的帮助。"
        #         "骨膜下反应性新骨产生明显，呈现层状反应，形成带有特征性的“葱皮样”现象。"
        #         "核素骨扫描除了能够明确肿瘤的部位和大小，也对肿瘤性质特别是是否存在肿瘤的转移和其他肿瘤性疾病的并存有帮助。"
        #         "骨髓穿刺检查可以排除其他骨髓内的恶性肿瘤活组织检查提供进一步的病理诊断依据。")

        data_all = []
        with open(path_patient + "/" + file, "r") as f:  # 打开文件
            data = f.read()  # 读取文件
            # print(data)
            for text in data.split("\n"):
                for sentence in text.split("。"):
                    # sen=sentence
                    for sen in sentence.split("，"):
                        if len(str(sen).split(" ")) >= 6:
                            for senterm in str(sen).split(" "):
                                if len(senterm) > 1:
                                    data_all.append(str(senterm))
                        else:
                            if len(sen) > 1:
                                data_all.append(str(sen))


        for sent in data_all:
            messages = []
            torch.cuda.empty_cache()
            # print(sent)
            messages_text = [{"role": "system",
                              "content": "请尽可能提取所有疾病症状实体，同时忽略其他类型的实体。每个实体一行，以'-'开头，无需指出实体类型"},
                             {"role": "user", "content": sent}]
            messages.append(messages_text[0])
            messages.append(messages_text[1])

            # print(messages)

            response = model.HuatuoChat(tokenizer, messages)

            # print(response)
            if str(response).startswith("请尽可能提取所有疾病症状实体"):
                continue
            else:
                print(response)
                with open(path_result + "/" + file_name, "a") as f:
                    f.write(response)  # 自带文件关闭功能，不需要再写f.close()
                    f.write("\n")
