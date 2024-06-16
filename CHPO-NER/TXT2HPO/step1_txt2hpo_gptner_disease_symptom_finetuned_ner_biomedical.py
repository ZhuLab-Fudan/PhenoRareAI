import sys
import txt2hpo_rank_gptner_disease_symptom_finetuned_ner_biomedical as txt2hpo
import pickle

import os

import pandas as pd

from scipy import spatial


import json



import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

datacohort='_gptner_disease_symptom_finetuned_ner_biomedical'
# raredisease="LOPD"



with open("bert-base-chinese-finetuned-NER-biomedical_feature_chpo_phenopro_core_dict.json") as fp:
    chinese_feature_chpo= json.load(fp)



tokenizer_chinese = AutoTokenizer.from_pretrained("bert-base-chinese-finetuned-NER-biomedical")

model_chinese = AutoModel.from_pretrained("bert-base-chinese-finetuned-NER-biomedical")



REALPATH = ''
chpo_dic_dir = REALPATH + 'src/chpo_202110.txt'
common_texts=[]
fi = open(chpo_dic_dir)
for line in fi:
    seq = line.strip().split('\n')
    hpoid = seq[0]
    common_texts.append(hpoid)

f=open(chpo_dic_dir)
documents_list=[]
for line in f:
    documents_list.append(line.strip().split("\n")[0])



StopWordtmp = ['。', '、', '~', '/', '，', '！', '？', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '｛', '｝', '-', '－', '～', '［', '］', '〔', '〕', '．', '＠', '￥', '•', '.']

class HPO_Class:
    def __init__(self, _id=[], _name=[], _alt_id=[], _def=[], _comment=[], _synonym=[], _xref=[], _is_a=[], _alt_Hs={},
                 _chpo=[], _chpo_def=[]):
        self._id = _id
        self._name = _name
        self._alt_id = _alt_id
        self._def = _def
        self._comment = _comment
        self._synonym = _synonym
        self._xref = _xref
        self._is_a = _is_a
        self._father = set()
        self._child_self = set()
        self._alt_Hs = _alt_Hs
        self._chpo = _chpo
        self._chpo_def = _chpo_def



REALPATH = ''

HPOs = txt2hpo.loading(REALPATH + '/src/HPOdata.pk')



chpo_dic_dir = REALPATH + 'src/chpo_202110.txt'
split_punc_dir = REALPATH + 'src/split_punc.txt'
rm_dir = REALPATH + 'src/rmwords.txt'
# rm_pro_dir=REALPATH+'src/rmwords_pro.txt'
mapping_list_dir = REALPATH + 'src/mapping_list_202110.txt'

filter_list = []
data_filter = pd.read_excel("raredisease_termidname_20221215hpo_202110chpo.xlsx")
for i in data_filter["termid"]:
    filter_list.append(i)

path_data="./hospital_data"
import os

path = "gpt_result/finetuned_ner_biomedical_disease_symptom_result"
os.makedirs(path)

path_gpt_result="gpt_result/finetuned_ner_biomedical_disease_symptom_result"


patient_already_list=[]
filessss = os.listdir(path_gpt_result)
for filea in filessss:
    patient_name_a=str(filea).split(".")[0]
    patient_already_list.append(patient_name_a)


files = os.listdir(path_data)
for file in files:
    patient_name=str(file).split(".")[0]
    if patient_name not in patient_already_list:


        elements = []
        elements_index_place_begin = []
        elements_index_place_end = []
        path_allgram = "."
        data_allgram = pd.read_csv(path_allgram + "/" + "gptner_disease_symptom_elements.csv")
        # patient_fill,allngram_text,index,index_place,label,score
        for i in range(0, data_allgram.shape[0]):

            patient_allgramtext = data_allgram["gptner_enetity"][i]
            patient_idmap = data_allgram["patient_name"][i]
            patient_beginid = data_allgram["begin_id"][i]
            patient_endid = data_allgram["end_id"][i]
            # text_begin= data_allgram["beginid"][i]
            # text_end= data_allgram["endid"][i]
            # patient_index_place = data_allgram["index_place"][i]
            if str(patient_name) == str(patient_idmap):
                elements.append(str(patient_allgramtext))
                elements_index_place_begin.append(patient_beginid)
                elements_index_place_end.append(patient_endid)
        # print()

        hpos = txt2hpo.mapping(elements, mapping_list_dir, HPOs)


        with open("chinese_feature_elements"+datacohort+".json") as fp:
            chinese_feature_elements = json.load(fp)


        old = set()
        given_hpos = []
        for i in range(0, len(elements)):
            if hpos[i] != ['None']:
                # fo.write(elements[i] + '\t' + ','.join(hpos[i]) + '\n')
                for one in hpos[i]:
                    if one not in old:
                        old.add(one)
                        given_hpos.append([one, i])


        fo_csv = pd.DataFrame()
        Given_HPO_list = []
        HPO_name_list = []
        HPO_name_cn_list = []
        Element_list = []
        score_list = []
        beginplace_list=[]
        endplace_list=[]
        # fo_csv.write('#Given_HPO\tHPO_name\tHPO_name_cn\tElement\tscore\n')
        # i = 0
        for i in range(0,len(given_hpos)):
        # while i < len(given_hpos):
            hpo = given_hpos[i][0]
            element = elements[given_hpos[i][1]]
            beginplaceid=elements_index_place_begin[given_hpos[i][1]]
            endplaceid = elements_index_place_end[given_hpos[i][1]]
            if len(HPOs[hpo]._chpo) > 0:
                chpo = HPOs[hpo]._chpo[0]
            else:
                chpo = '无'

            term1=str(element)
            term2 = str(chpo)


            ####chinese
            chinese_feature_vectors_1 = chinese_feature_elements[term1]
            if term2 in chinese_feature_chpo:# and chpo != '无':

                chinese_feature_vectors_2 = chinese_feature_chpo[term2]


            else:
                all_names = []
                all_names.append(term2)
                # bs = 128  # batch size during inference
                all_embs = []
                for i in tqdm(np.arange(0, len(all_names))):
                    toks = tokenizer_chinese.batch_encode_plus(all_names[i:i + len(all_names)],
                                                       padding="max_length",
                                                       max_length=25,
                                                       truncation=True,
                                                       return_tensors="pt")
                    toks_cuda = {}
                    # for k, v in toks.items():
                    #     toks_cuda[k] = v.cuda()
                    for k, v in toks.items():
                        toks_cuda[k] = v
                    cls_rep = model_chinese(**toks_cuda)[0][:, 0, :]  # use CLS representation as the embedding
                    all_embs.append(cls_rep.cpu().detach().numpy())
                all_embs = np.concatenate(all_embs, axis=0)
                chinese_feature_vectors_2=np.array(all_embs)[0].tolist()


            score_chinese = 1 - spatial.distance.cosine(chinese_feature_vectors_1, chinese_feature_vectors_2)


            score = score_chinese

            Given_HPO_list.append(hpo)
            HPO_name_list.append(HPOs[hpo]._name[0])
            HPO_name_cn_list.append(chpo)
            Element_list.append(element)
            score_list.append(score)
            beginplace_list.append(beginplaceid)
            endplace_list.append(endplaceid)
            # fo.write(hpo + '\t' + HPOs[hpo]._name[0] + '\t' + chpo + '\t' + element + '\t' + str(score) + '\n')

            # i += 1

        fo_csv["Given_HPO"] = Given_HPO_list
        fo_csv["HPO_name"] = HPO_name_list
        fo_csv["HPO_name_cn"] = HPO_name_cn_list
        fo_csv["Element"] = Element_list
        fo_csv["begin"] = beginplace_list
        fo_csv["end"] = endplace_list
        fo_csv["score"] = score_list
        #
        fo_csv = fo_csv.sort_values(by=["score"], ascending=False)

        fo_csv.reset_index(inplace=True)
        #
        # fo.write('#Given_HPO\tHPO_name\tHPO_name_cn\tElement\tbegin\tend\tscore\n')
        fo_csv.to_csv(path_gpt_result+"/"+patient_name+"-rank.csv",index=None)





