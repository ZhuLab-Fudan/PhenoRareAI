#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from collections import defaultdict

import os
import json


for ictype in ["BASE_diseaseIC"]:
    path_patient = "BASE_IC_adddelete_result"

    path_finally = ""

    files_patient_folder = os.listdir(path_patient)

    patient2disease_similarity_score = defaultdict(dict)
    for file in files_patient_folder:
        file1 = str(file)
        patient_name_str = str(file1).split(".")[0]
        # if patient_name_str.startswith("No2"):
        similarity_matrix_new = defaultdict(dict)

        patient_df = pd.read_csv(path_patient + "/" + file1)
        for i in range(0, patient_df.shape[0]):
            disease = patient_df["disease"][i]
            score = patient_df["score"][i]
            similarity_matrix_new[patient_name_str][disease] = score

        for patient in similarity_matrix_new:
            for disease in similarity_matrix_new[patient]:
                patient2disease_similarity_score[patient][disease] = similarity_matrix_new[patient][disease]

    with open(path_finally + "/" + ictype+"_adddelete_hospital_result.json",
              'w') as fp:
        json.dump(patient2disease_similarity_score, fp, indent=2)

    # files_patient_folders = os.listdir(path_patient)
    #
    # patient2disease_similarity_score = defaultdict(dict)
    # for files in files_patient_folders:
    #     file2 = str(files)
    #     patient_name_str = str(file2).split(".")[0]
    #     if patient_name_str.startswith("No1"):
    #         similarity_matrix_new = defaultdict(dict)
    #
    #         patient_df = pd.read_csv(path_patient + "/" + file2)
    #         for i in range(0, patient_df.shape[0]):
    #             disease = patient_df["disease"][i]
    #             score = patient_df["score"][i]
    #             similarity_matrix_new[patient_name_str][disease] = score
    #
    #         for patient in similarity_matrix_new:
    #             for disease in similarity_matrix_new[patient]:
    #                 patient2disease_similarity_score[patient][disease] = similarity_matrix_new[patient][disease]
    #
    # with open(path_finally + "/" + ictype+"_doccano_LOPD_adddelete_result.json",
    #           'w') as fp:
    #     json.dump(patient2disease_similarity_score, fp, indent=2)
