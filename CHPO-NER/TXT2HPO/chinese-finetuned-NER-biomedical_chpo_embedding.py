import sys
import pickle
import os


import json




from transformers import AutoTokenizer, AutoModel

tokenizer_clinicalbert = AutoTokenizer.from_pretrained("bert-base-chinese-finetuned-NER-biomedical")

model_clinicalbert = AutoModel.from_pretrained("bert-base-chinese-finetuned-NER-biomedical")





N=0

# instructor_feature_list_df={}
# pubmedbert_feature_list_df={}
clinicalbert_feature_list_df={}



path = "src"

fi = open(path + '/' + 'HPOdata.pk', 'rb')
HPOs = pickle.load(fi)

HPO_all_set=set()
for HPO in HPOs:
    check_list = HPOs[HPO]._chpo
    for term in check_list:
        HPO_all_set.add(term)

REALPATH = ''


chpo_dic_dir = REALPATH + 'src/chpo_202110.txt'

fi = open(chpo_dic_dir)
for line in fi:
    seq = line.rstrip().split('\t')[0]
    HPO_all_set.add(seq)
fi.close()

for term in HPO_all_set:

    N = N + 1


    # term = "患者15年前无明显诱因出现四肢乏力"

    # sentences_term = list(term)
    inputs = tokenizer_clinicalbert(term, return_tensors="pt", padding=True, add_special_tokens=False)

    outputs = model_clinicalbert(**inputs)
    #####最后一层
    pooler_output = outputs.pooler_output

    # print('---pooler_output: ', pooler_output.tolist()[0])

    embedding_list = pooler_output.tolist()[0]


    clinicalbert_feature_list_df.update({term: embedding_list})



path_result=""


with open(path_result+"/"+"bert-base-chinese-finetuned-NER-biomedical_feature_chpo_phenopro_core_dict.json", 'w',encoding= 'utf-8') as fp:
    json.dump(clinicalbert_feature_list_df, fp, ensure_ascii=False,indent=2)

