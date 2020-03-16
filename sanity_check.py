import unittest
from model import MANN
import gensim
import numpy as np
from Data_Process import *
import torch


def load_word2vec_matrix(w2vpath, embedding_size):
    '''
    @description: read the pretrained word2vec model
    @param w2vpath{string}:the path of word2vec model
    @param embedding_size{int}:the size of word embedding vectors
    @return: vector(word embedding vectors),wvmodel(dict of word2index),vocab_size
    '''
    word2vec_file = w2vpath
    wvmodel = {}
    if os.path.isfile(word2vec_file):
        model = gensim.models.Word2Vec.load(word2vec_file)
        vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
        vocab_size = len(vocab)
        print(vocab_size)
        vector = np.zeros([vocab_size+1, embedding_size])
        for key, value in vocab.items():
            wvmodel[key] = value+1
            if len(key) > 0:
                vector[value+1] = model[key]
        vector[0] = np.random.uniform(-5.0, 5.0, embedding_size)
        return vector, wvmodel, vocab_size


class MANNCheck(unittest.TestCase):
    def forward_check(self):
        filepath = '../data'
        embedding_size = 100
        image_emb_size = 100
        domain_emb_size = 100
        print('***********load_word2vec***********')
        word2vec_matrix, wvmodel, vocab_size = load_word2vec_matrix(
            os.path.join(filepath, 'word2vec_' + str(embedding_size), 'w2vmodel'), embedding_size)
        config = {"image_height": 64,
                  "image_width": 64,
                  "max_images_num": 10,
                  "max_domains_num": 10,
                  "max_text_len": 300,
                  "datapath": filepath,
                  "all_domains_num": 1047,
                  "batch_size": 128,
                  "vocab_size": vocab_size,
                  "input_size": 100,
                  "image_emb_size": 100,
                  "domain_emb_size": 100,
                  "keep_prob": 0.8,
                  # the score margin between positive example and negative examples.
                  "margin": 0.5,
                  'l2_reg': 0.00004,
                  'score_layer_size1': 200
                  }
        images = {}
        images_emb_map = {}
        traindata = Data(config=config, images=images,
                         images_emb_map=images_emb_map, is_train=True)
        traindata.load_domains(os.path.join(
            filepath, 'knowledge_num_list.txt'))
        traindata.reload_train_data_with_num(
            trainpath=os.path.join(filepath, 'Train.json'), compare_neg_num=2)
        batch_num = traindata.batch_num
        traindata.shuffle()

        model = MANN(word2vec_matrix,
                word_embedding_dim=embedding_size,
                all_domains_num=config['all_domains_num'],
                max_domains_num=config['max_domains_num'],
                domain_embedding_dim=domain_emb_size,
                hidden_size=domain_emb_size + embedding_size,
                max_sentence_length=config['max_text_len'])

        batch_no = 0
        while not traindata.end:
            _, qid_materials, posqid_materials, negqid_materials, domains_mat, data_label = traindata.next_batch_NoImages(
                    wvmodel=wvmodel)
            score_positive, score_negitive = model.forward(
                qid_materials, posqid_materials, negqid_materials, domains_mat)
            loss = torch.abs(score_positive - score_negitive).sum()
            loss.backward()
            batch_no += 1
        


if __name__ == "__main__":
    unittest.TextTestRunner().run(MANNCheck("forward_check"))
