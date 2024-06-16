
import os
import json

from collections import defaultdict
from functools import reduce
import numpy as np
from collections import defaultdict
import pandas as pd
import json
import pickle
import os
import string



import re

path_1="CBERT-NER-API/bert_base_chinese_finetuned_biomedical_nerresult"
path_result_1="bert_base_chinese_finetuned_biomedical_result"

path_2="CBERT-NER-API/bert_base_chinese_medical_nerresult"
path_result_2="bert_base_chinese_medical_result"



files = os.listdir(path_1)

# #######bert_base_chinese_finetuned_biomedical_lopdsma
data_ele=pd.DataFrame()
patient_list=[]
word_list=[]
begin_list=[]
end_list=[]
entity_group_name_list=[]
for file in files:
    patient_name=str(file).split(".")[0]

    with open(path_1 + "/" + patient_name + ".json") as fp:
        patient_ner = json.load(fp)

    for k in range(0, len(patient_ner)):
        term = patient_ner[k]

        fo = open(path_result_1+ "/" + patient_name, 'a')
        if str(term["entity_group"]).startswith("B_"):
            beginid = term["start"]
            endid = term["end"]
            score = float(term["score"])
            entity_group_name = str(term["entity_group"]).split("_")[-1]
            word_part = str(term["word"]).replace(' ', '')
            for i in range(k + 1, len(patient_ner)):
                if str(patient_ner[i]["entity_group"]).startswith("I_"):
                    endid = int(patient_ner[i]["end"])
                    word_part = word_part + str(patient_ner[i]["word"]).replace(' ', '')
                else:
                    break

            patient_list.append(patient_name)
            word_list.append(word_part)
            begin_list.append(beginid)
            end_list.append(endid)
            entity_group_name_list.append(entity_group_name)

            fo.write(str(word_part) + '\t' + str(beginid) + '\t' + str(endid) + '\t' + str(entity_group_name) + '\n')

        else:
            continue
data_ele["patient"]=patient_list
data_ele["word"]=word_list
data_ele["beginid"]=begin_list
data_ele["endid"]=end_list
data_ele["entity_group_name"]=entity_group_name_list
data_ele.to_csv("./TXT2HPO/bert_base_chinese_finetuned_biomedical_elements.csv",index=None)




filess = os.listdir(path_2)

#########bert_base_chinese_medical_lopdsma
data_ele=pd.DataFrame()
patient_list=[]
word_list=[]
begin_list=[]
end_list=[]
entity_group_name_list=[]

for file in filess:
    patient_name=str(file).split(".")[0]

    with open(path_2 + "/" + patient_name + ".json") as fp:
        patient_ner = json.load(fp)

    for k in range(0, len(patient_ner)):
        term = patient_ner[k]

        fo = open(path_result_2 + "/" + patient_name, 'a')

        if str(term["entity_group"])=="M":
            beginid = term["start"]
            endid = term["end"]
            score = float(term["score"])
            entity_group_name = str(term["entity_group"])
            word_part = str(term["word"]).replace(' ', '')

            for i in range(k + 1, len(patient_ner)):
                if str(patient_ner[i]["word"]).startswith("#"):
                    word_part = word_part + str(patient_ner[i]["word"]).replace('#', '')
                else:
                    break

            fo.write(str(word_part) + '\t' + str(beginid) + '\t' + str(endid) + '\t' + str(entity_group_name) + '\n')

            patient_list.append(patient_name)
            word_list.append(word_part)
            begin_list.append(beginid)
            end_list.append(endid)
            entity_group_name_list.append(entity_group_name)

        else:
            continue

data_ele["patient"]=patient_list
data_ele["word"]=word_list
data_ele["beginid"]=begin_list
data_ele["endid"]=end_list
data_ele["entity_group_name"]=entity_group_name_list
data_ele.to_csv("./TXT2HPO/bert_base_chinese_medical_elements.csv",index=None)
