import sys
import pickle
import os
import pandas as pd
import numpy as np


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

    while i < (len(target)-len(pattern)):#+1):

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
        if char_ == c:
            return True
    return False



REALPATH = ''

mapping_list_dir = REALPATH + 'src/mapping_list_202110.txt'


output_path = REALPATH + 'output_sunday'

mapping_dict = {}
# mapping_list=[]
fi = open(mapping_list_dir)
for line in fi:
    seq = line.rstrip().split('\t')
    if seq[0] not in mapping_dict:
        # mapping_list.append(seq[0])
        mapping_dict[seq[0]] = [seq[1]]
    else:
        if seq[1] not in mapping_dict[seq[0]]:
            mapping_dict[seq[0]].append(seq[1])
fi.close()



path_corpus = "./hospital_data"

files = os.listdir(path_corpus)
for file in files:
    file_name=str(file)
    patient_name = str(file).split(".")[0]
    input_dir = path_corpus + "/" + file_name
    # output_dir = output_path + "/" + patient_name
    # fo = open(output_dir, 'w')
    patient_result_csv=pd.DataFrame()
    element_list=[]
    hpo_list=[]
    begin_list=[]
    end_list=[]
    # fo.write('#Givern_term\n')
    # fo.write(open(input_dir).read().replace('\n', ' ').replace('\r', ' '))
    doucument_ner = open(input_dir).read().replace('\n', ' ').replace('\r', ' ')

    # fo.write('#Interpreted_term\tHPOs\tPlace\n')
    old = set()
    given_hpos = []
    i = 0
    tqdmlenth = 0

    ##################################

    # doucument_ner=doucument_ner.strip(".")
    for termmap in mapping_dict:
        if len(termmap) > 1:
            # print("doucument_ner",doucument_ner)
            # print("termmap", termmap)
            findkeylist = sunday_match(target=doucument_ner, pattern=termmap)
            if len(findkeylist) > 0:
                for beginid in findkeylist:
                    endid = beginid + len(termmap)
                    for hpo in mapping_dict[termmap]:
                        element_list.append(str(termmap))
                        hpo_list.append(hpo)
                        begin_list.append(beginid)
                        end_list.append(endid)
                        # fo.write(termmap + '\t' + hpo + '\t' + '[' + str(beginid) + ':' + str(endid) + ']' + '\n')

    patient_result_csv["element"]=element_list
    patient_result_csv["hpo"]=hpo_list
    patient_result_csv["begin"]=begin_list
    patient_result_csv["end"]=end_list
    patient_result_csv.to_csv(output_path+"/"+patient_name+".csv",index=None)








