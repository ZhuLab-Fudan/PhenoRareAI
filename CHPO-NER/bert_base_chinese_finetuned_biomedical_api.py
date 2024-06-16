import requests
import os
import time
import pandas as pd
import numpy as np
import json


API_TOKEN=""
API_URL = "https://api-inference.huggingface.co/models/Adapting/bert-base-chinese-finetuned-NER-biomedical"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

modelname="bert_base_chinese_finetuned_biomedical"
path_hospital_json="/hospital_data_json"###transfer EHR data to json file

path_result="CBERT-NER-API/"+modelname

files = os.listdir(path_hospital_json)
for file in files:
    patient_name=str(file).split(".")[0]
    with open(path_hospital_json+"/"+file) as fp:
        patient_text = json.load(fp)
    time.sleep(5)
    requests.adapters.DEFAULT_RETRIES = 5
    try:
        output = query(patient_text)
    except requests.exceptions.ConnectionError:
        continue

    # output1=output.split["["][-1]
    # output2 = output1.split["]"][0]
    result_json = output
    if "error" not in result_json:
        with open(path_result + "/" + patient_name + ".json", "w", encoding='utf-8') as f:
            # json.dump(dict_, f)  # 写为一行
            json.dump(result_json, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行

        # os.remove(path_hospital_json+"/"+file)
