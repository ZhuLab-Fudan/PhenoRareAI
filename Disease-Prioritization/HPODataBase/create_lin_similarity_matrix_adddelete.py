#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split protein list into three parts: train set, ltr set, and test set.
We will get three annotation datasets, three protein lists, and term list.
Besides, we will split HPO terms into several groups according to frequency.
"""

from sklearn.preprocessing import MultiLabelBinarizer

import math

from ontology import HumanPhenotypeOntology

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
# ontology_t4 = HumanPhenotypeOntology(config["ontology"]["time4"]["path"],
#                                      version=config["ontology"]["time4"]["version"])
#
# global variable, ancestors of each HPO term
ancestors = dict()
# global variable, frequency of terms
freq = None
# global variable, information content of HPO terms
ic = None




def lin_sim(x):
    """
    Lin measure, see Lin D. An information-theoretic definition of
    similarity. In: ICML, vol. Vol. 98, no. 1998; 1998. p. 296–304.
    :param x: tuple of index name, i.e. (row_term, col_term)
    :return: similarity
    """
    global ancestors
    global ic

    term_a, term_b = x
    # set values on the diagonal to 1
    if term_a == term_b:
        return 1
    # ancestors of term_a
    ancestors_a = ancestors[term_a]
    # ancestors of term_b
    ancestors_b = ancestors[term_b]
    # all common ancestors of term_a and term_b (and also in terms)
    common_ancestors = list(ancestors_a & ancestors_b)
    # information content of most informative common ancestor
    ic_mica = ic[common_ancestors].max()
    # similarity between term_a and term_b
    sim = 2 * ic_mica / (ic[term_a] + ic[term_b])
    return sim




path2022="20221215"

with open(path2022+"/"+"disease2hpo_adddelete.json") as fp:
    new_annotation = json.load(fp)



propagated_annotation = dict()
for disease in new_annotation:
    propagated_annotation[disease] = list(
        ontology_t3.transfer(new_annotation[disease]))
        # - {get_root()} -set(get_subontology(ontology_t2.version)))


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

term_list_sets=set(term_list)

for term in term_list_sets:
    ancestors[term] = ontology_t3.get_ancestors([term])
                      # - {get_root()} -set(get_subontology(ontology_t2.version))

similarity = pd.DataFrame(0, index=term_list_sets, columns=term_list_sets)
similarity = similarity.stack()
similarity.loc[:] = similarity.index.map(lin_sim)

similarity = similarity.unstack()
# write to the json file
similarity = similarity.to_dict(orient="index")
# total_num=len(common_terms)


with open(path2022+"/"+"lin_diseaseic_similarity_matrix20221215_adddelete.json", 'w') as fp:
    json.dump(similarity, fp, indent=2)
