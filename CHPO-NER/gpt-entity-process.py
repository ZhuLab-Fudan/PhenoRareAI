 #!/usr/bin/env python

import numpy as np
import pandas as pd


import json


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



######chinese

path="GPT-NER-API"

####"label_gptner_disease_symptom.json"
for cohort in ["LOPD"]:
    for prompt in ["disease_symptom"]:#["symptom"]:#["disease","biomedical","entity"]:

        data_gpt_ner = pd.DataFrame()
        patient_name_list = []
        gptner_enetity_list = []
        begin_id_list = []
        end_id_list = []

        with open(path + "/" + "label_"+cohort + "gptner"+"_"+prompt+".json") as fp:
            datajson = json.load(fp)

        data_map = pd.read_csv("./patient_id.csv")
        # data["patinet_id,idmapN"]
        #
        for idx in datajson:
            patient_name = str(data_map["patinet_id"][int(idx)])
            patient_name_ori=patient_name#.split("_")[0]
            patient_text = datajson[idx]["text"]
            patient_response = datajson[idx]["response"]
            # str(patient_response).replace("['", "").replace("']", "").replace("\n-", "---")
            # for i in patient_response:
            #     print(i)

            # fo = open(path_result+"/"+patient_name, 'w')

            for termfind in patient_response:
                for term_ene in str(termfind).split("\n"):
                    enetity = term_ene.strip(" ")[1:]
                    enetity.strip(" ")
                    enetity.strip(";")
                    enetity.strip(",")
                    enetity.strip(".").strip(";").strip(",").strip(" ").strip()

                    if len(enetity) > 1:
                        begin_list = []
                        end_list = []
                        findlist = sunday_match(target=patient_text, pattern=enetity)
                        if len(findlist) == 0:
                            continue
                            # print("findno")
                            # begin_list.append("None")
                            # end_list.append("None")
                        else:
                            for id in findlist:
                                beginid = id
                                endid = beginid + len(enetity)
                                # patient_name_list.append(patient_name)
                                patient_name_list.append(patient_name_ori)
                                gptner_enetity_list.append(enetity)
                                begin_id_list.append(beginid)
                                end_id_list.append(endid)

        data_gpt_ner["patient_name"] = patient_name_list
        data_gpt_ner["gptner_enetity"] = gptner_enetity_list
        data_gpt_ner["begin_id"] = begin_id_list
        data_gpt_ner["end_id"] = end_id_list

        df_new = data_gpt_ner.drop_duplicates()

        df_new.to_csv(cohort.lower() + "_gpt_" + prompt + ".csv", index=None)
