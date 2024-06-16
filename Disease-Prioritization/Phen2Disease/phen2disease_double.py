#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split protein list into three parts: train set, ltr set, and test set.
We will get three annotation datasets, three protein lists, and term list.
Besides, we will split HPO terms into several groups according to frequency.
"""
import json
import pandas as pd
import pickle
from collections import defaultdict
from functools import reduce
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import json
import pickle
from multiprocessing import Pool
# import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import math
from collections import defaultdict
from functools import reduce
from ontology import HumanPhenotypeOntology
from ontology import get_root, get_subontology
import os
from collections import defaultdict
import json
import pickle
from functools import reduce
import numpy as np
import pandas as pd


#
with open("split_dataset_lableler.json") as fp:
    config= json.load(fp)
# load various versions of HPO
# ontology_t0 = HumanPhenotypeOntology(config["ontology"]["time0"]["path"],
#                                      version=config["ontology"]["time0"]["version"])
# ontology_t1 = HumanPhenotypeOntology(config["ontology"]["time1"]["path"],
#                                      version=config["ontology"]["time1"]["version"])
# ontology_t2 = HumanPhenotypeOntology(config["ontology"]["time2"]["path"],
#                                      version=config["ontology"]["time2"]["version"])
ontology_t3 = HumanPhenotypeOntology(config["ontology"]["time3"]["path"],
                                     version=config["ontology"]["time3"]["version"])



# global variable, ancestors of each HPO term
ancestors = dict()
# global variable, frequency of terms
freq = None
# global variable, information content of HPO terms
ic = None


matrixtype="lin"


path2022="./HPODataBase/20221215"



with open(path2022+"/"+"disease2hpo.json") as fp:
    new_annotation = json.load(fp)


propagated_annotation = dict()
for disease in new_annotation:
    propagated_annotation[disease] = list(
        ontology_t3.transfer(new_annotation[disease]))
        # - {get_root()} -set(get_subontology(ontology_t3.version)))


propagated_annotation_new = defaultdict(set)
for disease in propagated_annotation:
    for term in propagated_annotation[disease]:
        propagated_annotation_new[term].add(disease)

test_dataset=propagated_annotation_new

term_list = list(test_dataset.keys())

disease_list= set(reduce(lambda a, b: set(a) | set(b),
                          test_dataset.values()))

mlb = MultiLabelBinarizer()
df_test_dataset = pd.DataFrame(mlb.fit_transform(test_dataset.values()),
                               columns=mlb.classes_,
                               index=test_dataset.keys()).reindex(
                               columns=disease_list, index=term_list, fill_value=0).transpose()

test_annotation = df_test_dataset.reindex(
        index=disease_list, columns=term_list, fill_value=0)


test_annotation = test_annotation.loc[:, (test_annotation != 0).any(axis=0)]
# remove rows containing only zeros
test_annotation = test_annotation[(test_annotation.T != 0).any()]


total_disease = len(test_annotation.index)
# sum over the diseases to calculate the frequency of terms
freq = test_annotation.sum(axis=0)/total_disease
# information content of each HPO term
ic = -freq.apply(math.log2)



########################################################################################
path_disease = "./HPODataBase/20221215/diseaselist"

path_patient="./HPODataBase/Hospital_DATA"
path_single="double"




#读取path路径下的全部文件名
files_disease_folder = os.listdir(path_disease)
files_patient_folder = os.listdir(path_patient)

term_list_sets=set(term_list)

for term in term_list_sets:
    ancestors[term] = ontology_t3.get_ancestors([term])
                      # - {get_root()} -set(get_subontology(ontology_t3.version))

# similarity = pd.DataFrame(0, index=term_list_sets, columns=term_list_sets)
# similarity = similarity.stack()
# similarity.loc[:] = similarity.index.map(lin_sim)
#
# similarity = similarity.unstack()
# # write to the json file
# similarity = similarity.to_dict(orient="index")
# # total_num=len(common_terms)


#########inheritance set
sp_term="HP:0000118"
inheritance_list=[]
inheritance_list_new=[]
root_set={get_root()}
for term in root_set:
    inheritance_list.append(term)
sub_root_set=set(get_subontology(ontology_t3.version))-{sp_term}
for term in sub_root_set:
    inheritance_list.append(term)
for term in sub_root_set:
    for term_d in ontology_t3.get_descendants([term]):
        inheritance_list.append(term_d)

inheritance_list=set(inheritance_list)
for term in inheritance_list:
    inheritance_list_new.append(term)

#########root set
ic_term_subontology_list=[]
term_subontology_list=[]
for term in get_subontology(ontology_t3.version):
    term_subontology_list.append(term)
for term in term_subontology_list:
    if term=='HP:0032223':
        continue
    if term=='HP:0032443':
        continue
    if term=='HP:0040279':
        continue
    else:
        ic_term_subontology_list.append(ic[term])
epsilon=max(ic_term_subontology_list)
########read similarity

path_similarity = "./HPODataBase/20221215"

with open(path_similarity+"/"+matrixtype+"_similarity_matrix20221215.json") as fp:
    similarity = json.load(fp)




for file in files_patient_folder:
    file1= str(file)
    patient_name_str = str(file1)
    termlist_patient=pd.read_csv(path_patient+"/"+file1,header=None)

    patient2disease_similarity_score = defaultdict(dict)

    for file_compare in files_disease_folder:
        disease_name_str = str(file_compare)
        file2 = str(file_compare)
        termlist_disease = pd.read_csv(path_disease + "/" + file2, header=None)

        disease_term_list_ORI = list(termlist_disease[0].values)
        patient_term_list_ORI = list(termlist_patient[0].values)
        ################################################
        ##排除不存在的term
        disease_term_list = []
        for term in disease_term_list_ORI:
            if term in term_list:
                disease_term_list.append(term)
        disease_term_list = list(disease_term_list)
        ################################################
        patient_term_list = []
        for term in patient_term_list_ORI:
            if term in term_list:
                patient_term_list.append(term)
        patient_term_list = list(patient_term_list)
        ################################################
        ################################################
        ####Filter
        disease_term_list_filter = []
        patient_term_list_filter = []
        # disease_term_list
        # patient_term_list
        for term in disease_term_list:

            judgment = 0
            if term in inheritance_list:
                judgment = judgment + 1
            term_disease_filter_temp_list = []
            for term_d in ontology_t3.get_descendants([term]):
                term_disease_filter_temp_list.append(term_d)
            for term_f in disease_term_list:
                if term_f in term_disease_filter_temp_list:
                    judgment = judgment + 1
                else:
                    judgment = judgment + 0
            if judgment == 0:
                disease_term_list_filter.append(term)

        for term in patient_term_list:

            judgment = 0
            if term in inheritance_list:
                judgment = judgment + 1
            term_patient_filter_temp_list = []
            for term_d in ontology_t3.get_descendants([term]):
                term_patient_filter_temp_list.append(term_d)
            for term_f in patient_term_list:
                if term_f in term_patient_filter_temp_list:
                    judgment = judgment + 1
                else:
                    judgment = judgment + 0
            if judgment == 0:
                patient_term_list_filter.append(term)
        ################################################
        max_score = 0
        term_weight = []
        #####################
        ######double weight
        #####################
        for term_1 in patient_term_list_filter:
            score_term_list = []
            for term_2 in disease_term_list_filter:
                score_term_list.append(similarity[term_1][term_2])

            if np.array(score_term_list).shape[0] == 0:
                    max_score = max_score + 0
            else:
                term_weight.append(ic[term_1])
                max_score = max_score + max(score_term_list) * ic[term_1]


        for term_1 in disease_term_list_filter:
            score_term_list = []
            for term_2 in patient_term_list_filter:
                score_term_list.append(similarity[term_1][term_2])

            if np.array(score_term_list).shape[0] == 0:
                max_score = max_score + 0
            else:
                term_weight.append(ic[term_1])
                max_score = max_score + max(score_term_list) * ic[term_1]


        if sum(term_weight) == 0:
            max_score = max_score / (epsilon)
        else:
            max_score = max_score / sum(term_weight)
        patient2disease_similarity_score[patient_name_str][disease_name_str]=max_score

    with open(path_single+"/"+patient_name_str+".json", 'w') as fp:
        json.dump(patient2disease_similarity_score, fp, indent=2)

