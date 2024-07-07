

from scipy import spatial

import json



import numpy as np

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


datacohort='_huatuogpt2ner_disease_symptom_lopd_medical_ner_later_1'

chinese_dict_path="/public/home/zhaiwq/tmp/HPONEREL_pycharm_project/GPUNER/S-DATA"

with open(chinese_dict_path+"/"+"bert-base-chinese-medical-ner_feature_chpo_phenopro_core_dict.json") as fp:
    chinese_feature_chpo= json.load(fp)

# with open("pubmedbert_feature_chpo_phenopro_core_dict.json") as fp:
#     pubmedbert_feature_chpo= json.load(fp)
# with open("instructor_feature_chpo_phenopro_core_dict.json") as fp:
#     instructor_feature_chpo= json.load(fp)
#
#
# with open("doc2vec_feature_chpo_phenopro_core_dict.json") as fp:
#     doc2vec_feature_chpo= json.load(fp)
#
# model_doc2vec = Word2Vec.load("doc2vec_chpo.model")


# tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
# model_pubmedbert = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")#.cuda()
tokenizer_chinese = AutoTokenizer.from_pretrained("/public/home/zhaiwq/tmp/HPONEREL_pycharm_project/roberta-cpt-medical-ner/medical_ner_model/bert-base-chinese-medical-ner")

model_chinese = AutoModel.from_pretrained("/public/home/zhaiwq/tmp/HPONEREL_pycharm_project/roberta-cpt-medical-ner/medical_ner_model/bert-base-chinese-medical-ner")


# tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
# model_pubmedbert = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")#.cuda()

# model_instructor = INSTRUCTOR('hkunlp/instructor-large')
#
# tfidf_vec = TfidfVectorizer()

# rouge = Rouge()

REALPATH = '/public/home/zhaiwq/tmp/HPONEREL_pycharm_project/PhenoPro-master/TXT2HPO/'

chpo_dic_dir = REALPATH + 'src/chpo_202110.txt'


# f=open(chpo_dic_dir)
# documents_list=[]
# for line in f:
#     documents_list.append(line.strip().split("\n")[0])
#
#
# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents_list)]
# model_doc2vec = Doc2Vec(documents, vector_size=128, window=10, min_count=1, workers=64)  # 指定向量长度为5

# model_dbow = Doc2Vec(dm=0, min_count=1, window=10, vector_size=3000, epochs=30, workers=10)
# model_dbow.build_vocab(documents)
# model_dbow.train(documents, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

#
# import fasttext.util
# fasttext.util.download_model('zh', if_exists='ignore')
# model_fasttext = fasttext.load_model('cc.zh.300.bin')
#
# fasttext.util.reduce_model(model_fasttext, 128)
# model_fasttext.get_dimension()
# # print(model_fasttext.get_word_vector('罕见病'))


# ltp = LTP()

# ltp.init_dict(path=chpo_dic_dir, max_window=10)




class HPO_Class:
    def __init__(self, _id=[], _name=[], _alt_id=[], _def=[], _comment=[], _synonym=[], _xref=[], _is_a=[], _alt_Hs={},
                 _chpo=[], _chpo_def=[]):
        self._id = _id
        self._name = _name
        self._alt_id = _alt_id
        self._def = _def
        self._comment = _comment
        self._synonym = _synonym
        self._xref = _xref
        self._is_a = _is_a
        self._father = set()
        self._child_self = set()
        self._alt_Hs = _alt_Hs
        self._chpo = _chpo
        self._chpo_def = _chpo_def


#########################################################################################################################################################

def dumping(obo_file, chpo_file, save_file_dir):
    import pickle

    HPOs = {}
    _id = [];
    _name = [];
    _alt_id = [];
    _def = [];
    _comment = [];
    _synonym = [];
    _xref = [];
    _is_a = [];
    _chpo = [];
    _chpo_def = []
    fi = open(obo_file)
    obo_terms = fi.read().split('[Term]')
    fi.close()
    alt_Hs = {}
    for term in obo_terms:
        if 'id: ' in term:
            seq = term.split('\n')
            for one in seq:
                if ': ' in one:
                    if 'id: ' in one and 'alt_id: ' not in one:
                        _id.append(one.split(': ')[1])
                    if 'name: ' in one:
                        _name.append(one.split(': ')[1])
                    if 'alt_id: ' in one:
                        alt_Hs[one.split(': ')[1]] = _id[-1]
                        _alt_id.append(one.split(': ')[1])
                    if 'def: ' in one:
                        _def.append(one.split(': ')[1])
                    if 'comment: ' in one:
                        _comment.append(one.split(': ')[1])
                    if 'synonym: ' in one:
                        if '"' in one:
                            _synonym.append(one.split(': ')[1].split('"')[1])
                        else:
                            _synonym.append(one.split(': ')[1])
                    if 'xref: ' in one:
                        _xref.append(one.split(': ')[1])
                    if 'is_a: ' in one:
                        _is_a.append(one.split(': ')[1].split('!')[0].replace(' ', ''))

            HPOs[_id[0]] = HPO_Class(_id, _name, _alt_id, _def, _comment, _synonym, _xref, _is_a, _chpo=[],
                                     _chpo_def=[])
            _id = [];
            _name = [];
            _alt_id = [];
            _def = [];
            _comment = [];
            _synonym = [];
            _xref = [];
            _is_a = [];
            _chpo = [];
            _chpo_def = []

    # Alt_names
    HPOs['HP:0000118']._alt_Hs = alt_Hs

    # CHPO
    fi = open(chpo_file,encoding='gbk')
    for line in fi:
        seq = line.rstrip().split('\t')
        hpoid = seq[0]
        flag = 1
        if hpoid in HPOs:
            _id = hpoid
        elif hpoid in alt_Hs:
            # print(hpoid)
            _id = alt_Hs[hpoid]
        else:
            flag = 0
        if flag == 1:
            _chpo = seq[1]
            HPOs[_id]._chpo.append(_chpo)
            # print(HPOs[_id]._chpo)
            if len(seq) >= 3:
                _chpo_def = seq[2]
                # print(HPOs[_id]._chpo_def)
                HPOs[_id]._chpo_def.append(_chpo_def)
                # print(HPOs[_id]._chpo_def)

    # Find_father
    def find_father(_ori_id, _id):
        if HPOs[_id]._name == 'All':
            pass
        else:
            for one in HPOs[_id]._is_a:
                HPOs[_ori_id]._father.add(one)
                find_father(_ori_id, one)

    j = 0
    for _id in HPOs:
        find_father(_id, _id)
        j = j + 1;
        print(" Finding HPOs' ancestor nodes: " + str(j) + ' HPOs ', end='\r')

    # Find_children
    print('')
    j = 0
    for asHPO in HPOs:
        for HPO in HPOs:
            if asHPO in HPOs[HPO]._father:
                HPOs[asHPO]._child_self.add(HPO)
            HPOs[asHPO]._child_self.add(asHPO)  # add self
        j = j + 1;
        print(" Finding HPOs' child nodes: " + str(j) + ' HPOs ', end='\r')

    print('')
    fo = open(save_file_dir, 'wb')
    data = HPOs
    pickle.dump(data, fo)
    fo.close()


#########################################################################################################################################################

def loading(data_file):
    import pickle
    fi = open(data_file, 'rb')
    HPOs = pickle.load(fi)
    return HPOs


#########################################################################################################################################################

# def splitting(input_dir, HPOs, chpo_dic_dir,  split_punc_dir,  rm_en_dir, rm_cn_dir, rm_pro_dir, output_dir):
def splitting(input_dir, HPOs, chpo_dic_dir, split_punc_dir, rm_dir):
    import jieba
    import jieba.posseg as psg
    import nltk

    split_punc = set()
    fi = open(split_punc_dir)
    for line in fi:
        split_punc.add(line.rstrip())
    fi.close()

    words = open(input_dir).read().replace('\n', '; ').replace('\r', '; ')
    L = len(words)
    rmlist = open(rm_dir).read().rstrip().split('\n')

    # Txt2phrase
    phrases = []
    old = set()
    i = 0
    tmp = 0
    while i < len(words):
        word = words[i]
        if word in split_punc:
            phrase = words[tmp:i].strip()

            if phrase not in old:
                old.add(phrase)
                for one in rmlist:
                    phrase = phrase.replace(one, '')
                if len(phrase) > 0:
                    phrases.append(phrase)

            tmp = min(i + 1, L - 1)
        i += 1

    phrase = words[tmp:i].strip()

    if phrase not in old:
        old.add(phrase)
        for one in rmlist:
            phrase = phrase.replace(one, '')
        if len(phrase) > 0:
            phrases.append(phrase)

    #    print(phrases )

    # print(phrases)
    # Split phrase



    def split_cn(phrase):
        # ltp = LTP()
        # ltp.init_dict(path=chpo_dic_dir, max_window=10)

        # segment, _ = ltp.seg(list(phrase))
        # seq=segment[0]

        jieba.load_userdict(chpo_dic_dir)
        seq = jieba.cut(phrase)
        return seq

    # print(phrases)
    elements = []
    old = set()
    for phrase in phrases:
        flag = 'en'
        for alpha in phrase:
            if ord(alpha) > 255:
                flag = 'cn'
        if flag == 'cn':
            out = split_cn(phrase)
            seq = []
            seq.append(phrase)
            for one in out:
                if len(one) > 1:
                    seq.append(one)
        else:
            seq=" "
            # print(seq)
        for element in seq:
            if element not in old:
                old.add(element)
                elements.append(element)

    return elements


#########################################################################################################################################################

def mapping(elements, mapping_list_dir, HPOs):
    mapping_list = {}
    fi = open(mapping_list_dir)
    for line in fi:
        seq = line.rstrip().split('\t')
        if seq[0] not in mapping_list:
            mapping_list[seq[0]] = [seq[1]]
        else:
            if seq[1] not in mapping_list[seq[0]]:
                mapping_list[seq[0]].append(seq[1])
    fi.close()


    def compareterm_cn(term1,term2):

        # sentences_term1 = list(term1)
        # embeddings_term1 = model_instructor.encode(sentences_term1)
        # ###instructor
        # embeddings_term1 = instructor_feature_elements[term1]
        # if term2 in instructor_feature_chpo:
        #     embeddings_term2 = instructor_feature_chpo[term2]
        # else:
        #     embeddings_chpo = model_instructor.encode(list(term2))
        #     embeddings_term2 = embeddings_chpo[0].tolist()
        #
        # score_instructor = 1 - spatial.distance.cosine(embeddings_term1, embeddings_term2)


        # ####pubmedbert
        # pubmedbert_feature_vectors_1 = pubmedbert_feature_elements[term1]
        # if term2 in pubmedbert_feature_chpo:
        #     pubmedbert_feature_vectors_2 = pubmedbert_feature_chpo[term2]
        # else:
        #     all_names = []
        #     all_names.append(term2)
        #     # bs = 128  # batch size during inference
        #     all_embs = []
        #     for i in tqdm(np.arange(0, len(all_names))):
        #         toks = tokenizer.batch_encode_plus(all_names[i:i + len(all_names)],
        #                                            padding="max_length",
        #                                            max_length=25,
        #                                            truncation=True,
        #                                            return_tensors="pt")
        #         toks_cuda = {}
        #         # for k, v in toks.items():
        #         #     toks_cuda[k] = v.cuda()
        #         for k, v in toks.items():
        #             toks_cuda[k] = v
        #         cls_rep = model_pubmedbert(**toks_cuda)[0][:, 0, :]  # use CLS representation as the embedding
        #         all_embs.append(cls_rep.cpu().detach().numpy())
        #     all_embs = np.concatenate(all_embs, axis=0)
        #     pubmedbert_feature_vectors_2=np.array(all_embs)[0].tolist()
        #
        #
        # score_pubmedbert = 1 - spatial.distance.cosine(pubmedbert_feature_vectors_1, pubmedbert_feature_vectors_2)


        ####chinese
        chinese_feature_vectors_1 = chinese_feature_elements[term1]
        if term2 in chinese_feature_chpo:
            chinese_feature_vectors_2 = chinese_feature_chpo[term2]
        else:
            all_names = []
            all_names.append(term2)
            # bs = 128  # batch size during inference
            all_embs = []
            for i in tqdm(np.arange(0, len(all_names))):
                toks = tokenizer_chinese.batch_encode_plus(all_names[i:i + len(all_names)],
                                                   padding="max_length",
                                                   max_length=25,
                                                   truncation=True,
                                                   return_tensors="pt")
                toks_cuda = {}
                # for k, v in toks.items():
                #     toks_cuda[k] = v.cuda()
                for k, v in toks.items():
                    toks_cuda[k] = v
                cls_rep = model_chinese(**toks_cuda)[0][:, 0, :]  # use CLS representation as the embedding
                all_embs.append(cls_rep.cpu().detach().numpy())
            all_embs = np.concatenate(all_embs, axis=0)
            chinese_feature_vectors_2=np.array(all_embs)[0].tolist()


        score_chinese = 1 - spatial.distance.cosine(chinese_feature_vectors_1, chinese_feature_vectors_2)


        # ####doc2vec
        # vectord1 = doc2vec_feature_elements[term1]
        # if term2 in doc2vec_feature_chpo:
        #     vectord2  = doc2vec_feature_chpo[term2]
        # else:
        #     vectord2 = model_doc2vec.infer_vector(list(term2))
        #
        # score_doc2vec = 1 - spatial.distance.cosine(vectord1, vectord2)
        #
        #
        #
        # ######score_match
        # score_Levenshtein = Levenshtein.ratio(''.join(term1).replace(" ", ""), ''.join(term2).replace(" ", ""))
        # score_difflib = difflib.SequenceMatcher(None, ''.join(term1).replace(" ", ""),
        #                                         ''.join(term2).replace(" ", "")).quick_ratio()
        #
        # reference_1 = ''.join(term1).replace(" ", "")
        # reference_2 = ''.join(term1).replace(" ", "")
        #
        # try:
        #     scores = rouge.get_scores(reference_1, reference_2)
        #     score_rouge = scores[0]['rouge-1']["r"]
        # except:
        #     score_rouge = 0

        # score=(score_spatial+score_Levenshtein+score_difflib+score_rouge)/4

        # score = 1 - (1 - score_Levenshtein) \
        #         * (1 - score_difflib) \
        #         * (1 - score_rouge) * (1 - score_instructor) * (1 - score_pubmedbert)

        # score = 1 - (1 - score_Levenshtein) * (1 - score_difflib)*(1 - score_instructor) * (1 - score_pubmedbert)
        # score = score_Levenshtein + score_difflib + score_rouge+score_instructor+score_pubmedbert+score_doc2vec
        score=score_chinese

        return score


    def interpreting_cn(keywords):
        keywords = keywords
        if keywords == '':
            return ['None']
        # if len(keywords) == 1:
        #     return ['None']
        score = []
        for HPO in HPOs:
            tmp = [0]
            check_list = HPOs[HPO]._chpo
            for term in check_list:
                if len(keywords) == 1:
                    tmp.append(0)
                else:
                    tmp.append(compareterm_cn(keywords, term))
            if 'HP:0000118' in HPOs[HPO]._father:
                score.append([max(tmp), HPO])
        score.sort(reverse=True)
        # if keywords=='语言发育障碍':
        #    print(score)

        return score

    def purifyHPO(score):
        output = []
        final_output = []
        if float(score[0][0]) < 0.001:
            ok_final_output = ['None']
        else:
            tmp = score[0][0]
            i = 0
            ori_output = []
            while score[i][0] == tmp:
                ori_output.append(score[i][1])
                i = i + 1
            child_group = set()
            for HPO in ori_output:
                this_HPO = set()
                this_HPO.add(HPO)
                tmp = HPOs[HPO]._child_self - this_HPO
                child_group = child_group | tmp
            for one in ori_output:
                if one not in child_group:
                    output.append(one)

            limit_length = 1

            if len(output) > limit_length:

                tmp = []
                for one in output:
                    father_len = len(HPOs[one]._father)
                    tmp.append([father_len / float(father_len + len(HPOs[one]._child_self)), one])
                    # tmp.append([father_len/float(father_len),one])
                tmp.sort()

                for one in tmp[0:limit_length]:
                    final_output.append(one[1])


                if len(tmp) > limit_length:
                    combined_father = set()
                    this_HPO = set()

                    this_HPO.add(tmp[limit_length][1])
                    combined_father = HPOs[tmp[limit_length][1]]._father | this_HPO
                    for one in tmp[limit_length:]:
                        this_HPO = set()
                        this_HPO.add(one[1])
                        combined_father = combined_father & (HPOs[one[1]]._father | this_HPO)
                    tmptmp = []
                    for one in combined_father:
                        tmptmp.append([len(HPOs[one]._father), one])
                    tmptmp.sort()
                    ####################20171108
                    try:
                        final_output.append(tmptmp[-1][1])
                    except Exception as e:
                        pass

                # if len(tmp) > limit_length:
                    iii = limit_length
                    while iii < len(tmp) and tmp[iii][0] == tmp[limit_length - 1][0]:
                        final_output.append(tmp[iii][1])
                        iii += 1
                    limit_length = iii

                # print(final_output)



                tmp = []
                for one in final_output:
                    if 'HP:0000118' not in HPOs[one]._child_self:
                        tmp.append(one)
                final_output = tmp
            else:
                final_output = output

            ok_final_output = []
            for one in final_output:
                if 'HP:0000118' in HPOs[one]._father:
                    ok_final_output.append(one)
            if len(ok_final_output) == 0:
                ok_final_output = ['None']

        return ok_final_output

    # def purifyHPO(score):
    #     ok_final_output=[]
    #     if float(score[0][0]) < 0.0005:
    #         ok_final_output = ['None']
    #     else:
    #         final_output = []
    #         final_output.append(score[0][1])
    #         # final_output.append(score[1][1])
    #
    #         for one in final_output:
    #             if 'HP:0000118' in HPOs[one]._father:
    #                 ok_final_output.append(one)
    #         if len(ok_final_output) == 0:
    #             ok_final_output = ['None']
    #     return ok_final_output


    def mapping_cn(element):
        score = interpreting_cn(element)
        hpos = purifyHPO(score)
        return hpos

    alt_Hs = HPOs['HP:0000118']._alt_Hs
    mapped_hpos = []

    instructor_feature_elements_df = {}
    pubmedbert_feature_elements_df = {}
    doc2vec_feature_elements_df={}
    chinese_feature_elements_df = {}
    for element in elements:
        sentences_element = list(element)
        element = str(element)
        # ####instructor
        # embeddings_element = model_instructor.encode(sentences_element)
        # instructor_feature_vectors = embeddings_element

        ####pubmedbert
        # all_names = []
        # all_names.append(element)
        # # bs = 128  # batch size during inference
        # all_embs = []
        # for i in tqdm(np.arange(0, len(all_names))):
        #     toks = tokenizer.batch_encode_plus(all_names[i:i + len(all_names)],
        #                                        padding="max_length",
        #                                        max_length=25,
        #                                        truncation=True,
        #                                        return_tensors="pt")
        #     toks_cuda = {}
        #     # for k, v in toks.items():
        #     #     toks_cuda[k] = v.cuda()
        #     for k, v in toks.items():
        #         toks_cuda[k] = v
        #     cls_rep = model_pubmedbert(**toks_cuda)[0][:, 0, :]  # use CLS representation as the embedding
        #     all_embs.append(cls_rep.cpu().detach().numpy())
        # all_embs = np.concatenate(all_embs, axis=0)
        # pubmedbert_feature_vectors = np.array(all_embs)  # .tolist()[0]
        #
        #
        ####chinese

        all_names = []
        all_names.append(element)
        # bs = 128  # batch size during inference
        all_embs = []
        for i in tqdm(np.arange(0, len(all_names))):
            toks = tokenizer_chinese.batch_encode_plus(all_names[i:i + len(all_names)],
                                               padding="max_length",
                                               max_length=25,
                                               truncation=True,
                                               return_tensors="pt")
            toks_cuda = {}
            # for k, v in toks.items():
            #     toks_cuda[k] = v.cuda()
            for k, v in toks.items():
                toks_cuda[k] = v
            cls_rep = model_chinese(**toks_cuda)[0][:, 0, :]  # use CLS representation as the embedding
            all_embs.append(cls_rep.cpu().detach().numpy())
        all_embs = np.concatenate(all_embs, axis=0)
        chinese_feature_vectors=np.array(all_embs)[0].tolist()


        # ####doc2vec
        # doc2vec_feature_vectors = model_doc2vec.infer_vector(sentences_element)


        # instructor_feature_elements_df.update({element: instructor_feature_vectors[0].tolist()})
        # pubmedbert_feature_elements_df.update({element: pubmedbert_feature_vectors[0].tolist()})
        # doc2vec_feature_elements_df.update({element: doc2vec_feature_vectors.tolist()})
        chinese_feature_elements_df.update({element: chinese_feature_vectors})

    # with open("instructor_feature_elements_lopd"+datacohort+".json", 'w', encoding='utf-8') as fp:
    #     json.dump(instructor_feature_elements_df, fp, ensure_ascii=False, indent=2)

    # with open("pubmedbert_feature_elements_lopd"+datacohort+".json", 'w', encoding='utf-8') as fp:
    #     json.dump(pubmedbert_feature_elements_df, fp, ensure_ascii=False, indent=2)

    # with open("doc2vec_feature_elements_lopd"+datacohort+".json", 'w', encoding='utf-8') as fp:
    #     json.dump(doc2vec_feature_elements_df, fp, ensure_ascii=False, indent=2)
    with open("chinese_feature_elements_lopd"+datacohort+".json", 'w', encoding='utf-8') as fp:
        json.dump(chinese_feature_elements_df, fp, ensure_ascii=False, indent=2)


    for element in elements:
        if element.upper() in mapping_list:
            mapped_hpos.append(mapping_list[element])
        elif element.upper() in HPOs:
            mapped_hpos.append([element])
        elif element.upper() in alt_Hs:
            mapped_hpos.append([alt_Hs[element]])
        else:
            flag = 'en'
            for alpha in element:
                if ord(alpha) > 255:
                    flag = 'cn'
            if flag == 'en':
                mapped_hpos.append(['None'])
                # if len(element) <= 2:
                #     mapped_hpos.append(['None'])
                # else:
                #     hpos = mapping_en(element)
                #     mapped_hpos.append(hpos)
            else:

                with open("chinese_feature_elements_lopd"+datacohort+".json") as fp:
                    chinese_feature_elements = json.load(fp)

                # with open("instructor_feature_elements_lopd"+datacohort+".json") as fp:
                #     instructor_feature_elements = json.load(fp)
                #
                # with open("doc2vec_feature_elements_lopd"+datacohort+".json") as fp:
                #     doc2vec_feature_elements = json.load(fp)


                hpos = mapping_cn(element)
                mapped_hpos.append(hpos)

    #    i=1
    #    while i<len(elements):
    #        print(elements[i])
    #        print(mapped_hpos[i])

    #       i+=1
    return mapped_hpos



