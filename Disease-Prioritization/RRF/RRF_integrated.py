
import os
import json

from collections import defaultdict
from functools import reduce
import numpy as np
from collections import defaultdict
import pandas as pd
import json
import pickle


eps=0.00001

# "phen2disease_diseaseic_hospital_lin__integrated_sum_result.json"
# "BASE_diseaseIC_hospital_result.json"
# "Phrank_hospital_result.json"
# "LIRICAL_hospital_result.json"
# for model_name in ["LOPD","SMA"]:  #

with open("phen2disease_diseaseic_hospital_lin_integrated_sum_result") as fp:
    phen2disease_result = json.load(fp)

with open("Phrank_hospital_result.json") as fp:
    phrank_result = json.load(fp)

with open("LIRICAL_hospital_result.json") as fp:
    lirical_result = json.load(fp)

with open("BASE_diseaseIC_hospital_result.json") as fp:
    baseic_result = json.load(fp)

disease_list = set(reduce(lambda a, b: set(a) | set(b),
                          phen2disease_result.values()))
disease_phrank_list = set(reduce(lambda a, b: set(a) | set(b),
                          phrank_result.values()))
disease_lirical_list = set(reduce(lambda a, b: set(a) | set(b),
                                 lirical_result.values()))
score_full = max(len(disease_list),len(disease_phrank_list),len(disease_lirical_list))

phen2disease_zscore = defaultdict(dict)

for patient in phen2disease_result:
    # print("1_1")
    patient_score_list = []
    for gene in phen2disease_result[patient]:
        patient_score_list.append(float(phen2disease_result[patient][gene]))
    patient_score_max = np.array(patient_score_list).max()
    patient_score_min = np.array(patient_score_list).min()
    for gene in phen2disease_result[patient]:
        phen2disease_zscore[patient][gene] = float(
            (float(phen2disease_result[patient][gene]) - patient_score_min) / (
                        patient_score_max - patient_score_min))

phrank_zscore = defaultdict(dict)

for patient in phrank_result:
    # print("1_2")
    patient_score_list = []
    for gene in phrank_result[patient]:
        patient_score_list.append(float(phrank_result[patient][gene]))
    patient_score_max = np.array(patient_score_list).max()
    patient_score_min = np.array(patient_score_list).min()
    for gene in phrank_result[patient]:
        phrank_zscore[patient][gene] = float(
            (float(phrank_result[patient][gene]) - patient_score_min) / (patient_score_max - patient_score_min))

baseic_zscore = defaultdict(dict)

for patient in baseic_result:
    # print("1_3")
    patient_score_list = []
    for gene in baseic_result[patient]:
        patient_score_list.append(float(baseic_result[patient][gene]))
    patient_score_max = np.array(patient_score_list).max()
    patient_score_min = np.array(patient_score_list).min()
    for gene in baseic_result[patient]:
        baseic_zscore[patient][gene] = float(
            (float(baseic_result[patient][gene]) - patient_score_min) / (patient_score_max - patient_score_min))



lirical_zscore = defaultdict(dict)
lirical_zscore_rank = defaultdict(dict)

for patient in lirical_result:
    # print("1_4")
    patient_score_rank = pd.DataFrame()
    patient_gene_list = []
    patient_score_list = []
    patient_rank_list = []
    for gene in lirical_result[patient]:
        patient_gene_list.append(gene)
        patient_score_list.append(float(lirical_result[patient][gene]))
    patient_score_rank["gene"] = patient_gene_list
    patient_score_rank["score"] = patient_score_list

    patient_score_rank.sort_values(by=["score"], axis=0, ascending=False, inplace=True)

    for i in range(0, patient_score_rank.shape[0]):
        patient_rank_list.append(score_full - i)

    patient_score_rank["rankscore"] = patient_rank_list

    for k in range(0, patient_score_rank.shape[0]):
        lirical_zscore_rank[patient][patient_score_rank["gene"][k]] = float(patient_score_rank["rankscore"][k])

for patient in lirical_zscore_rank:
    # print("1_3")
    patient_score_list = []
    for gene in lirical_zscore_rank[patient]:
        patient_score_list.append(float(lirical_zscore_rank[patient][gene]))
    patient_score_max = np.array(patient_score_list).max()
    patient_score_min = np.array(patient_score_list).min()
    for gene in lirical_zscore_rank[patient]:
        lirical_zscore[patient][gene] = float(
            (float(lirical_zscore_rank[patient][gene]) - patient_score_min) / (patient_score_max - patient_score_min))



phen2disease_zscore_integrated_mrr = defaultdict(dict)

for patient in phen2disease_zscore:
    for gene in phen2disease_zscore[patient]:
        if gene not in phrank_zscore[patient]:
            temp_list = []
            for genei in phrank_zscore[patient]:
                temp_list.append(float(phrank_zscore[patient][genei]))
            patient_score_min = np.array(temp_list).min()
            phrank_zscore[patient][gene] = patient_score_min

        if gene not in baseic_zscore[patient]:
            temp_list = []
            for genei in baseic_zscore[patient]:
                temp_list.append(float(baseic_zscore[patient][genei]))
            patient_score_min = np.array(temp_list).min()
            baseic_zscore[patient][gene] = patient_score_min

        if gene not in lirical_zscore[patient]:
            temp_list = []
            for genei in lirical_zscore[patient]:
                temp_list.append(float(lirical_zscore[patient][genei]))
            patient_score_min = np.array(temp_list).min()
            lirical_zscore[patient][gene] = patient_score_min

        phen2disease_zscore_integrated_mrr[patient][gene] = 1 / (
                    (1 / (eps + phen2disease_zscore[patient][gene])) + (
                        1 / (eps + phrank_zscore[patient][gene])) + (
                                1 / (eps + baseic_zscore[patient][gene])) + (
                                1 / (eps + lirical_zscore[patient][gene])))

with open("RRF_integrated_hospital_result.json", 'w') as fp:
    json.dump(phen2disease_zscore_integrated_mrr, fp, indent=2)

