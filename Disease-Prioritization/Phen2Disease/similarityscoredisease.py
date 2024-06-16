#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split protein list into three parts: train set, ltr set, and test set.
We will get three annotation datasets, three protein lists, and term list.
Besides, we will split HPO terms into several groups according to frequency.
"""
import json

import os
from collections import defaultdict
import json
import pickle
from functools import reduce
import numpy as np
import pandas as pd
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



for insertion in ["lin"]:
    for ictype in ["diseaseic"]:#"diseasegeneic", "geneic","diseaseic"]:
        for dataset in ["double", "patient"]:  # ,"disease"]:
            path_base = dataset

            path_finally = "diseasefinally"
            ########################################################################################

            # for model_name in ["LOPD"]:
            path_patient_score = path_base  # +"/"+str(model_name)
            files_folder = os.listdir(path_patient_score)

            similarity_matrix_combine = defaultdict(dict)
            for file1 in files_folder:
                file_str = str(file1)
                patient_name_str = str(file_str)
                patient_name_judge = str(file1).split(".")[0]
                # if patient_name_judge.startswith("No1"):
                similarity_matrix_new = defaultdict(dict)

                with open("%s/%s" % (path_patient_score, file1)) as fp:
                    similarity_matrix = json.load(fp)

                for patient in similarity_matrix:
                    for disease in similarity_matrix[patient]:
                        similarity_matrix_new[patient][disease] = 0
                ##############
                #####MAX
                ##############
                for patient in similarity_matrix_new:
                    for disease in similarity_matrix_new[patient]:
                        similarity_matrix_new[patient][disease] = similarity_matrix[patient][disease]

                for patient in similarity_matrix_new:
                    for disease in similarity_matrix_new[patient]:
                        if len(str(similarity_matrix_new[patient][disease])) < 5:
                            similarity_matrix_combine[patient][disease] = 0
                        else:
                            similarity_matrix_combine[patient][disease] = similarity_matrix_new[patient][
                                disease]

                with open(path_finally + "/" + "phen2disease_" + dataset + "_"+ictype +"_hospital_" + str(insertion) + "_result.json",
                          'w') as fp:
                    json.dump(similarity_matrix_combine, fp, indent=2)

