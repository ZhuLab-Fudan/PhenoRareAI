#!/bin/python3

import pickle
from functools import reduce
from collections import defaultdict
import pandas as pd
import json
import pickle
from multiprocessing import Pool
# import pandas as pd
import numpy as np

from phrank import Phrank
from phrank import utils as phrank_utils
import json
from collections import defaultdict
import json

from collections import defaultdict

import os

#设置待处理数据的路径
path="./HPODataBase/Hospital_DATA"

import os

path = "hospital_adddelete_result"
os.makedirs(path)

outpath = "hospital_adddelete_result"


#读取path路径下的全部文件名
files = os.listdir(path)
# s= []
#cmd = os.system("flirt -in 101b0.gz -ref 104b0.gz -dof 12 -out 101to104 -omat 101to104.mat")


DAG="data/hpodag_20221215.txt"
# DISEASE_TO_PHENO="data/disease_to_pheno.build127.txt"
# DISEASE_TO_GENE="data/gene_to_disease.build127.txt"
# GENE_TO_PHENO="data/gene_to_pheno.amelie.txt"
# DISEASE_TO_PHENO="data/disease_to_pheno_utf82022.txt"
# DISEASE_TO_GENE="data/gene_to_disease_utf82022.txt"
# GENE_TO_PHENO="data/gene_to_pheno_utf82022.txt"
DISEASE_TO_PHENO="data/disease_to_pheno_utf820221215_transfer_adddelete.txt"
DISEASE_TO_GENE="data/gene_to_disease_utf820221215_adddelete.txt"
GENE_TO_PHENO="data/gene_to_pheno_utf820221215_transfer.txt"
p_hpo = Phrank(DAG, diseaseannotationsfile=DISEASE_TO_PHENO, diseasegenefile=DISEASE_TO_GENE)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create HPO annotations without propagation from raw file.

Output format:
{ protein_id1: [ hpo_term1, hpo_term2, ... ],
  protein_id2: [ hpo_term1, hpo_term2, ... ],
  ...
}
"""

# defining patient genes and phenotypes
gene_set = set()
with open("data/genecard_list.txt") as fp:
    for line in fp:
        ensg = line.split('\n')[0]
        gene_set.add(ensg)

patient_genes = gene_set

for file in files:
    file1 = str(file)
    #取文件名的前三个字符用于重命名
    name_str = str(file1)

    term_set = set()
    with open("%s/%s"%(path,file1)) as fp:
        for line in fp:
            term = line.split('\n')[0]
            term_set.add(term)

    phenotypeset = term_set
    #['HP:0000077', 'HP:0030765', 'HP:0012115', 'HP:0002088', 'HP:0002099', 'HP:0001945', 'HP:0000719']

    patient_phenotypes = phenotypeset

    # sorting the genes by best match
    disease_ranking = p_hpo.rank_diseases(patient_genes, patient_phenotypes)
    # gene_ranking = p_hpo.rank_genes(patient_genes, patient_phenotypes)
    # print ("\nGene ranking")
    ensembl_disease_id = []
    similarity_score = []

    for disease_info in disease_ranking:
        ensembl_disease_id.append(disease_info[1])
        similarity_score.append(disease_info[0])
        # print ("ensembl gene id: %s\tsimilarity score: %.2f"%(gene_info[1],gene_info[0]))

    Disease_ranking_result = pd.DataFrame({'ensembl_disease_id': ensembl_disease_id,
                                        'similarity_score': similarity_score})

    Disease_ranking_result.to_csv('hospital_adddelete_result/%s'%(name_str),index=False,header=True)


path_1 = "hospital_adddelete_result"
filess = os.listdir(path_1)

annotation = defaultdict(dict)
#/share/inspurStorage/home1/zhaiwq/tmp/pycharm_project_515/phrank/demo/result
for file in filess:
    file1 = str(file)
    #取文件名的前三个字符用于重命名
    name_str = str(file1)
    # print(name_str)
    # if name_str[1]=="M":
    #     diease_str=name_str[0:4]+":"+name_str[4:]
    # else:
    #     diease_str=name_str[0:5]+":"+name_str[5:]
    patient_str=name_str

    # ensg2protein = ensg2uniprot("ENSG2GENEID.tab", ensg_column=0, uniprot_column=-4)

    with open("hospital_adddelete_result/%s"%(name_str)) as fp:
        for line in fp:
            if line.startswith('e'):
                continue
            disease,score= line.strip().split(',')
            # for protein_id in ensg2protein[ensg_id]:
            annotation[patient_str][disease] = float(score)


with open("Phrank_hospital_adddelete_result.json", 'w') as fp:
    json.dump(annotation, fp, indent=2)
