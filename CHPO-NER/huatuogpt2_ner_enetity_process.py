 #!/usr/bin/env python

import numpy as np
import pandas as pd


import Levenshtein

def sunday_match(target, pattern):
    """
    Sunday string match
    Args:
        target: the string that want to be searched
        pattern: the string that want to be judged

    Returns:
        positions where pattern appear in target
    """
    i = 0
    positions = []

    while i < (len(target)-len(pattern)):
        # print(i)


        if target[i:i+len(pattern)] == pattern:
            positions.append(i)
            i = i + 1
            # print("is  equal, jump 1 step, i: {}, char: {}".format(i, target[i]))
        # elif target[i+len(pattern)] not in pattern:
        elif not char_search(target[i+len(pattern)], pattern):
            i = i + len(pattern) + 1
            # print("not equal, jump n step, i: {}, char: {}".format(i, target[i]))
        else:
            i = i + 1
            # print("not equal, jump 1 step, i: {}, char: {}".format(i, target[i]))

    return positions


def char_search(char_, str_):
    """
    search if char_ in str_
    Args:
        char_: char
        str_: str

    Returns:
        if char_ in str_ return True, else False
    """
    for c in str_:
        # print(c)
        if char_ == c:
            return True
    return False



import os

path_txt ="HuatuoGPT2-NER-API"

foldersss = os.listdir(path_txt)

patient_list = []
for file in foldersss:
     file_name = str(file)
     patient_list.append(file_name)

head_list=['1','2','3','4','5','6','7','8','9','0','.','*','-']
StopWordtmp = ['*','-','。', '、', '~', '/', '，', '！', '？', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '｛', '｝', '-', '－', '～', '［', '］', '〔', '〕', '．', '＠', '￥', '•', '.']
StopWordtmp_little = ['。', '、', '，', '！', '？', '：', '-', '－','．','.']

split_all=''
for sp in StopWordtmp:
    split_all=split_all+'|'+sp

split_little=''
for sp in StopWordtmp_little:
    split_little=split_little+'|'+sp





######chinese
path=path_txt



for prompt in ["disease_disease"]:#["symptom"]:#["disease","biomedical","entity"]:

    data_gpt_ner = pd.DataFrame()
    patient_name_list = []
    gptner_enetity_list = []
    begin_id_list = []
    end_id_list = []


    #
    for patient in patient_list:



        text_path="/hospital_data"###EHR data

        patient_text = open(text_path + "/" + patient+".txt").read()  # .replace('\n', ' ')

        patient_text_split = open(text_path + "/" + patient + ".txt").read()  # .replace('\n', ' ')

        patient_text_list = []
        for sp in StopWordtmp:
            patient_text_split=patient_text_split.replace(sp, ';;;')
        for term in patient_text_split:
            patient_text_list.append(term.strip())

        allngram_list = []
        for winlength in [4, 13]:
            for term in patient_text_list:
                if len(term) < 4 or len(term) > 12:
                    if term not in allngram_list:
                        allngram_list.append(term)
                else:
                    for i in range(0, len(term) - 1):
                        termmap = term[i, i + winlength]
                        if termmap not in allngram_list:
                            allngram_list.append(termmap)

        patient_name = str(patient)

        with open(path + "/" + patient, "r") as myfile:
            for line in myfile:
                # line_ents=[]
                enetityall = line.split("\n")[0].strip()
                # enetityall.replace('.', '').replace('*', '').replace('-', '')
                enetityall = enetityall.strip()

                for sp in StopWordtmp:
                    enetityall=enetityall.replace(sp, ';;;')

                for enetity in enetityall.split(';;;'):
                    enetity=enetity.strip()

                    if not enetity.strip().isdigit() and 15>=len(enetity)>=2:
                        begin_list = []
                        end_list = []
                        findlist = sunday_match(target=patient_text, pattern=enetity)

                        if len(findlist) == 0:
                            findmaxmap_list=[]
                            findmaxmap_score_list = []
                            for termgram in allngram_list:
                                score=Levenshtein.ratio(enetity, termgram)
                                findmaxmap_score_list.append(score)
                            scoremax=max(findmaxmap_score_list)
                            if scoremax>=0.85:
                                for i in range(0, len(findmaxmap_score_list)):
                                    if float(findmaxmap_score_list[i]) == float(scoremax):
                                        findmaxmap_list.append(allngram_list[i])

                                for mapterm in findmaxmap_list:
                                    findlistmap = sunday_match(target=patient_text, pattern=mapterm)
                                    for id in findlistmap:
                                        beginid = id
                                        endid = beginid + len(enetity)
                                        # patient_name_list.append(patient_name)
                                        patient_name_list.append(patient_name)
                                        gptner_enetity_list.append(enetity)
                                        begin_id_list.append(beginid)
                                        end_id_list.append(endid)


                        else:
                            for id in findlist:
                                beginid = id
                                endid = beginid + len(enetity)
                                # patient_name_list.append(patient_name)
                                patient_name_list.append(patient_name)
                                gptner_enetity_list.append(enetity)
                                begin_id_list.append(beginid)
                                end_id_list.append(endid)



    data_gpt_ner["patient_name"] = patient_name_list
    data_gpt_ner["gptner_enetity"] = gptner_enetity_list
    data_gpt_ner["begin_id"] = begin_id_list
    data_gpt_ner["end_id"] = end_id_list

    df_new = data_gpt_ner.drop_duplicates()

    df_new.to_csv("./TXT2HPO/huatuogpt2ner_" + prompt + "_elements.csv", index=None)





