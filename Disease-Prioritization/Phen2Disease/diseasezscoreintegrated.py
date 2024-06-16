
import os
import json

from collections import defaultdict
from functools import reduce
import numpy as np
from collections import defaultdict
import pandas as pd
import json
import pickle

def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

for ictype in ["diseaseic"]:#"diseasegeneic", "geneic","diseaseic"]:
    path = "/public/home/zhaiwq/modelused/BASEGENEALL_pycharm_project/hospital/result/diseasefinally"

    for insertion in ["lin"]:
        # for model_name in ["LOPD"]:  # ["ic","jc","rel","graphic","lin","resnik"]:
        with open(path + "/" + "phen2disease_patient_"+ictype+"_hospital_" + str(insertion) + "_result.json") as fp:
            phen2disease_patient = json.load(fp)

        with open(path + "/" + "phen2disease_double_"+ictype+"_hospital_" + str(insertion)+ "_result.json") as fp:
            phen2disease_double = json.load(fp)

        # with open(path + "/" + "similarity_diseasealready_weight_diseaserankscore_"+model_name+"_result.json") as fp:
        #     phen2disease_disease = json.load(fp)

        # phen2disease_patient_zscore = defaultdict(dict)
        #
        # for patient in phen2disease_patient:
        #     patient_score_list = []
        #     for gene in phen2disease_patient[patient]:
        #         patient_score_list.append(float(phen2disease_patient[patient][gene]))
        #     patient_score_mean = np.array(patient_score_list).mean()
        #     patient_score_std = np.std(np.array(patient_score_list), ddof=1)
        #     for gene in phen2disease_patient[patient]:
        #         phen2disease_patient_zscore[patient][gene] = float(
        #             (float(phen2disease_patient[patient][gene]) - patient_score_mean) / patient_score_std)
        #
        #
        # phen2disease_double_zscore = defaultdict(dict)
        #
        # for patient in phen2disease_double:
        #     patient_score_list = []
        #     for gene in phen2disease_double[patient]:
        #         patient_score_list.append(float(phen2disease_double[patient][gene]))
        #     patient_score_mean = np.array(patient_score_list).mean()
        #     patient_score_std = np.std(np.array(patient_score_list), ddof=1)
        #     for gene in phen2disease_double[patient]:
        #         phen2disease_double_zscore[patient][gene] = float(
        #             (float(phen2disease_double[patient][gene]) - patient_score_mean) / patient_score_std)

        phen2disease_patient_zscore = phen2disease_patient
        phen2disease_double_zscore = phen2disease_double
        # # phen2disease_disease_zscore = phen2disease_disease
        #
        phen2disease_zscore_integrated_sum = defaultdict(dict)
        #
        for patient in phen2disease_double_zscore:
            for gene in phen2disease_double_zscore[patient]:
                # phen2disease_zscore_integrated_sum[patient][gene] = phen2disease_patient_zscore[patient][gene] + \
                #                                                     phen2disease_double_zscore[patient][gene] + \
                #                                                     phen2disease_disease_zscore[patient][gene]
                score = (phen2disease_patient_zscore[patient][gene] + phen2disease_double_zscore[patient][gene])
                # phen2disease_zscore_integrated_sum[patient][gene] = phen2disease_patient_zscore[patient][gene] + \
                #                                                     phen2disease_double_zscore[patient][gene]  # + \
                # if is_number(score):
                #     phen2disease_zscore_integrated_sum[patient][gene] = phen2disease_patient_zscore[patient][gene] + \
                #                                                         phen2disease_double_zscore[patient][gene]  # + \
                if len(str(score)) < 5:
                    phen2disease_zscore_integrated_sum[patient][gene] = 0
                else:
                    phen2disease_zscore_integrated_sum[patient][gene] = phen2disease_patient_zscore[patient][gene] + \
                                                                        phen2disease_double_zscore[patient][
                                                                            gene]  # + \

                # phen2disease_disease_zscore[patient][gene]
        phen2disease_zscore_integrated_max = defaultdict(dict)

        # for patient in phen2disease_patient_zscore:
        #     for gene in phen2disease_patient_zscore[patient]:
        #         # phen2disease_zscore_integrated_max[patient][gene] = max(phen2disease_patient_zscore[patient][gene],
        #         #                                                         phen2disease_double_zscore[patient][gene],
        #         #                                                         phen2disease_disease_zscore[patient][gene])
        #         phen2disease_zscore_integrated_max[patient][gene] = max(phen2disease_patient_zscore[patient][gene],
        #                                                                 phen2disease_double_zscore[patient][gene])

        with open(path + "/" + "phen2disease_"+ictype+"_hospital_" +str(insertion) + "_integrated_sum_result.json", 'w') as fp:
            json.dump(phen2disease_zscore_integrated_sum, fp, indent=2)
