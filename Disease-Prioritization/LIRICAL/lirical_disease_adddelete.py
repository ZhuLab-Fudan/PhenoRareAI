import json
import yaml
import math

import numpy as np
from collections import defaultdict

import os

from functools import reduce

from ontology import HumanPhenotypeOntology

with open("split_dataset_lableler.json") as fp:
    config= json.load(fp)
# load various versions of HPO
# ontology_t0 = HumanPhenotypeOntology(config["ontology"]["time0"]["path"],
#                                      version=config["ontology"]["time0"]["version"])
# ontology_t1 = HumanPhenotypeOntology(config["ontology"]["time1"]["path"],
#                                      version=config["ontology"]["time1"]["version"])
# ontology_t2 = HumanPhenotypeOntology(config["ontology"]["time1"]["path"],
#                                      version=config["ontology"]["time1"]["version"])
ontology_t3 = HumanPhenotypeOntology(config["ontology"]["time3"]["path"],
                                     version=config["ontology"]["time3"]["version"])


#设置待处理数据的路径

path="./HPODataBase/Hospital_DATA"

import os

path = "output_lirical"
os.makedirs(path)

path_out="output_lirical"
outputresult="output_lirical"#"phenopacket_outputpatient"
######读取path路径下的全部文件名
files = os.listdir(path)
# filesss=os.listdir(path_teshu)
for file in files:
    file1 = str(file)
    #取文件名的前三个字符用于重命名
    name_str = str(file1)
    with open("example_38_global2022.yml", 'r') as fp:
        exampleadjust = yaml.load(fp, Loader=yaml.FullLoader)
        # exampleadjust = yaml.load(fp)

    term_list = []
    with open("%s/%s"%(path,file1)) as fp:
        for line in fp:
            if line.split()[0] in ontology_t3:
                term_list.append(line.split()[0])

    for line in exampleadjust:
        if line == "hpoIds":
            exampleadjust[line] = term_list
        if line == "prefix":
            exampleadjust[line] = "%s"%(name_str)
        if line == "outdir":
            exampleadjust[line] = "%s/%s"%(outputresult,name_str)

    with open("experiment_lirical/%s.yml"%(name_str), "w") as f:
        yaml.dump(exampleadjust, f)

    os.system("java -jar LIRICAL2022.jar yaml -y experiment_lirical/%s.yml" %(name_str))

# for file in filesss:
#     file2 = str(file)
#     #取文件名的前三个字符用于重命名
#     name_str = str(file2)
#     with open("exampleori_38_global.yml", 'r') as fp:
#         exampleadjust = yaml.load(fp, Loader=yaml.FullLoader)
#         # exampleadjust = yaml.load(fp)
#
#     term_list = []
#     with open("%s/%s"%(path_teshu,file2)) as fp:
#         for line in fp:
#             if line.split()[0] in hpo_term:
#                 term_list.append(line.split()[0])
#
#     for line in exampleadjust:
#         if line == "hpoIds":
#             exampleadjust[line] = term_list
#         if line == "prefix":
#             exampleadjust[line] = "%s"%(name_str)
#         if line == "outdir":
#             exampleadjust[line] = "%s/%s"%(outputresult,name_str)
#
#     with open("simulation_experiment_patient/%s.yml"%(name_str), "w") as f:
#         yaml.dump(exampleadjust, f)
#
#     os.system("java -jar LIRICAL2021.jar yaml -y simulation_experiment_patient/%s.yml" %(name_str))

# for file in files:
#     file_str = str(file)
#     name_str=file_str[:-5]
#
#     os.system("java -jar LIRICAL2021.jar phenopacket -p %s/%s -d data --tsv -x %s -o %s" %(path,file_str,name_str,path_out))

# ###########################################################
# ###lricalscorerank
# ###########################################################

# # 设置待处理数据的路径

filess = os.listdir(path_out)
annotation = defaultdict(dict)

for file in filess:
    file_str = str(file)
    # 取文件名的前三个字符用于重命名
    # name_str = str(file_str)
    name_str = file_str#file1#file1[:-4]
    disease_str=name_str
    patient_str = disease_str
    # rank	diseaseName	diseaseCurie	pretestprob	posttestprob	compositeLR	entrezGeneId	variants

    with open("%s/%s/%s.tsv" % (path_out,patient_str,patient_str)) as fp:
        for line in fp:
            # annotation_tem = defaultdict(set)
            # annotation_tem_2 = defaultdict(set)
            # annotation_tem_3 = defaultdict(set)
            if line.startswith('!'):
                continue
            if line.startswith('r'):
                continue

            rank=line.strip().split('\t')[0]
            Rankscore = rank.replace(',', '')
            gene = line.strip().split('\t')[6]
            gene_id = gene.split(":")[-1]
            disease = line.strip().split('\t')[2]
            # omim_number = disease.split(':')[-1]
            number = line.split('\t')[5]
            logRscore = number.replace(',', '')

            annotation[patient_str][disease] = 0
            # if gene_id in geneid2card.keys():
            #     for genecard in geneid2card[gene_id]:
            #         annotation[disease_str][genecard] = 0
            #
            # if disease in disease2card.keys():
            #     for genecard in disease2card[disease]:
            #         annotation[disease_str][genecard] = 0
        #

    with open("%s/%s/%s.tsv" % (path_out,patient_str,patient_str)) as fp:
        for line in fp:
            # annotation_tem = defaultdict(set)
            # annotation_tem_2 = defaultdict(set)
            # annotation_tem_3 = defaultdict(set)
            if line.startswith('!'):
                continue
            if line.startswith('r'):
                continue

            rank = line.strip().split('\t')[0]
            Rankscore = rank.replace(',', '')
            gene = line.strip().split('\t')[6]
            gene_id = gene.split(":")[-1]
            disease = line.strip().split('\t')[2]
            # omim_number = disease.split(':')[-1]
            number = line.split('\t')[5]
            logRscore = number.replace(',', '')
            annotation[patient_str][disease] = annotation[patient_str][disease]+float(1/float(Rankscore))


with open("LIRICAL_hospital_adddelete_result.json", 'w') as fp:
    json.dump(annotation, fp, indent=2)


