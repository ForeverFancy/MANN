'''
@Author: Wang Xin
@Date: 2018-12-06 18:27:59
@LastEditTime: 2019-11-02 15:03:19
@Description: file content
'''
# -*- coding: utf-8 -*-
import os
import time
import sys
import datetime
import numpy as np
from gensim.models import Word2Vec
import json
import random
import math
from PIL import Image
import imghdr
import gensim
import pdb

class Data(object):
    def __init__(self,config,images, images_emb_map,is_train=True,is_case = False):
        '''
        @description: Data Class used to get batch and pass to the model
        @param config{dict}:a config dict which include all the configuration info. 
        E.g. max_text_len, batch_size, vocab_size, datapat..., 
        a detailed config can be found in the following test code.
        @param images{dict}:a dict that includes question id and the corresponding list of image ids
        @images_emb_map{dict}:a dict that maps the image id to an vector(which is computes by the pretrained auto-encoder)
        @is_train{bool}:if True, then create a Data class include training data. otherwise, include test data.
        @is_train{bool}:if True, load the case data(used to case study.)

        '''
        self.image_height=config['image_height']
        self.image_width=config['image_width']
        self.datapath =datapath= config['datapath']
        self.is_train = is_train
        self.batch_size = config['batch_size']
        self.image_embsize=config['image_emb_size']
        self.max_images_num = config['max_images_num']
        self.max_domains_num = config['max_domains_num']
        self.max_text_len = config['max_text_len']
        self.all_domains_num =config['all_domains_num']
        self.pos = 0
        self.end = False
        self.domains = {}
        self.qidset = set()
        self.images = images
        self.images_emb_map = images_emb_map
        self.images_mat ={}
        def load_train_data(trainpath):
            '''
            @description: load the test-pair from training file.
            @return: a list including the training pairs and labels
            '''

            data = []
            fin = open(trainpath, encoding='utf-8')
            for eachLine in fin:
                js = json.loads(eachLine)
                qid = js['qid']
                pair = js['pair']
                poslist = []
                neglist = []
                self.qidset.add(qid)
                for each in pair:
                    self.qidset.add(each[0])
                    #positive example
                    if int(each[1]) == 1: 
                        poslist.append(each[0])
                    #negative example
                    else:
                        neglist.append(each[0])
                poslen = len(poslist)
                neglen = min(len(neglist),10)
                for i in range(poslen):
                    for k in range(neglen):
                        if i == 0:
                            self.qidset.add(neglist[k])
                        data.append([qid, poslist[i], neglist[k], 1])

                for i in range(poslen):
                    negset = set()
                    num = 0
                    matchlen = min(10,neglen)
                    while num < matchlen:
                        k = random.randint(0, neglen-1)
                        if k not in negset:
                            negset.add(k)
                            data.append([qid,poslist[i],neglist[k],1])
                            num += 1
            fin.close()
            return  data

        def load_test_data(testpath):
            '''
            @description: load the test-pair from test file.
            @return: a list including the test pairs and labels
            '''
            data = []
            with open(testpath, encoding='utf-8') as fin:
                for eachLine in fin:
                    js = json.loads(eachLine)
                    qid = js['qid']
                    self.qidset.add(qid)
                    pair = js['pair']
                    for each in pair:
                        self.qidset.add(each[0])
                        if int(each[1]) == 1:
                            data.append([qid, each[0], each[0], 1])
                        else:
                            data.append([qid, each[0], each[0], 0])
            return data

        def load_case_data(datapath):
            '''
            @description: load the test-pair from case file.
            @return: a list including the case pairs and labels
            '''
            data = []
            fin = open(datapath, encoding='utf-8')
            for eachLine in fin:
                uuid = eachLine.replace('\n', '').split('\t')
                qid0 = uuid[0]
                qid1 = uuid[1]
                self.qidset.add(qid0)
                self.qidset.add(qid1)
                data.append([qid0, qid1, qid1, 1])
            fin.close()
            return data


        def load_qidfenci(wordpath):
            '''
            @description: load the tokenized question data
            @return: a dict 
            '''
            qidfenci = {}
            fin = open(wordpath)
            for eachLine in fin:
                js = json.loads(eachLine)
                qid = js['qid']
                fenci = js['fenci']
                qidfenci[qid] = fenci
            fin.close()
            return qidfenci


        #load case data
        if is_case:
            ldata = load_case_data(datapath + 'case.txt')
            self.data_size = len(ldata)
            self.data = np.array(ldata)
        else:
            #load training data
            if self.is_train:
                ldata = load_train_data(os.path.join(datapath,'Train.json'))
                self.data_size = len(ldata)
                self.data = np.array(ldata)
            #load test data
            else:
                ldata = load_test_data(os.path.join(datapath,'Test.json'))
                self.data_size = len(ldata)
                self.data = np.array(ldata)
        self.batch_num = int(math.ceil(float(self.data_size)/float(self.batch_size)))

        #load all the tokenized question data
        self.qid_fenci = load_qidfenci(os.path.join(datapath , 'qid_fenci_UseNum.json'))
    

    def load_domains(self,domainpath):
        '''
        @description: load the domain data
        @return: a dict that maps qid to the corresponding list of domains
        '''
        fin = open(domainpath)
        for eachLine in fin:
            pid = eachLine.replace('\n', '').split('\t')
            qid = pid[0]
            domainlist = []
            points = pid[1].split(' ')
            for eachpoint in points:
                domainlist.append(int(eachpoint))
            self.domains[qid] = domainlist
        fin.close()
    

    def reload_test_data(self, testpath): 
        data = []
        with open(testpath, encoding='utf-8') as fin:
            for eachLine in fin:
                js = json.loads(eachLine)
                qid = js['qid']
                self.qidset.add(qid)
                pair = js['pair']
                for each in pair:
                    self.qidset.add(each[0])
                    if int(each[1]) == 1:
                        data.append([qid, each[0], each[0], 1])
                    else:
                        data.append([qid, each[0], each[0], 0])
        self.data_size = len(data)
        self.batch_num = int(math.ceil(float(self.data_size) / float(self.batch_size)))
        self.data = np.array(data)



    def reload_train_data_with_num(self,trainpath = 'Train.json',compare_neg_num=2):
        '''
        @description: reload the training data with specified number of negtative example 
        @trainpath {string}:the path of training data 
        @compare_neg_num {int}:the negative example num for each question 
        '''
        data = []
        fin = open(trainpath, encoding='utf-8')
        for eachLine in fin:
            js = json.loads(eachLine)
            qid = js['qid']
            pair = js['pair']
            poslist = []
            neglist = []
            for each in pair:
                if int(each[1]) == 1:
                    poslist.append(each[0])
                else:
                    neglist.append(each[0])
            poslen = len(poslist)
            neglen = len(neglist)
            for i in range(poslen):
                negset = set()
                num = 0
                matchlen = min(compare_neg_num,neglen)
                while num < matchlen:
                    k = random.randint(0, neglen-1)
                    if k not in negset:
                        negset.add(k)
                        data.append([qid,poslist[i],neglist[k],1])
                        num += 1
        fin.close()
        self.data_size = len(data)
        self.batch_num = int(math.ceil(float(self.data_size) / float(self.batch_size)))
        self.data = np.array(data)


    def shuffle(self):
        self.pos = 0
        self.end = False
        shuffle_indices = np.random.permutation(np.arange(self.data_size))
        self.data = self.data[shuffle_indices]

    def reset(self):
        self.pos = 0
        self.end = False

    def image_open(self,im_path,dtype='float32'):
        '''
        @description: read an image into an array, resize and normalize it.
        '''
        x = (np.zeros((self.image_embsize, self.image_embsize))).astype(dtype)
        exist_image = False
        if os.path.exists(im_path)and imghdr.what(im_path):
            with Image.open(im_path) as im:
                im1 = im
                if im.mode == 'RGBA':
                    im_o = np.asarray(im)
                    im_c = np.subtract(255, im_o[:, :, 3])
                    if np.sum(im_o[:, :, 0:3]) == 0 or np.sum(np.subtract(255, im_o[:, :, 0:3])) == 0:
                        im1 = Image.fromarray(im_c)
                    else:
                        im1 = Image.fromarray(im_o[:, :, 0:3])
                '''
                The filter argument can be one of NEAREST (use nearest neighbour),
                 BILINEAR (linear interpolation in a 2x2 environment),
                 BICUBIC (cubic spline interpolation in a 4x4 environment),
                 or ANTIALIAS (a high-quality downsampling filter).
                 If omitted, or if the image has mode “1” or “P”,
                 it is set to NEAREST.
                '''
                im1 = im1.resize((self.image_height, self.image_width), Image.ANTIALIAS)
                imgray = im1.convert('L')
                im_array = np.asarray(imgray, dtype=np.float32)
                if im_array[0, 0] == 0:
                    im_array = np.subtract(255, im_array)

                im_array = im_array/ 255.0
                x = im_array
                exist_image = True
        return x,exist_image

    def image_create_mat(self,dtype='float32'):
        '''
        @description: build a dict which maps the Qid+ImgId to the pixel matrix.
        '''
        for eachqid in self.qidset:
            if eachqid in self.images:
                piclist = self.images[eachqid]
                llen = min(len(piclist), self.max_images_num)
                for idy in range(llen):
                    picpath = self.datapath + 'images/' + eachqid + '/' + piclist[idy]
                    im_array, exist_image = self.image_open(picpath)

                    if exist_image:
                        self.images_mat[picpath] = im_array

                    if not exist_image:
                        print(picpath)
                        print(exist_image)

    def image_convert(self,qidlist,dtype='float32'):
        '''
        @description: build the 3-D array(im_array) of image vectros and the corresponding mask vector(im_mask).
        im_array.shape= [qid,img_id,img_vec_size]
        im_mask.shape= [qid,img_id]
        '''
        nb_samples = len(qidlist)
        im_mask = (np.zeros((nb_samples, self.max_images_num))).astype(dtype)
        im_array = (np.zeros((nb_samples, self.max_images_num,self.image_embsize))).astype(dtype)
        for idx,eachqid in enumerate(qidlist):
            if eachqid in self.images:
                piclist = self.images[eachqid]
                llen = min(len(piclist),self.max_images_num)
                for idy in range(llen):
                    im_array[idx,idy] = self.images_emb_map[eachqid+'/'+piclist[idy]]
                    im_mask[idx,idy] = 1.0
        return im_array,im_mask



    def domains_convert(self,qidlist,dtype1='int32',dtype='float32'):
        nb_samples = len(qidlist)
        d_mask = (np.zeros((nb_samples, self.max_domains_num))).astype(dtype)
        d_array = (np.zeros((nb_samples, self.max_domains_num))).astype(dtype1)
        for idx, eachqid in enumerate(qidlist):
            if eachqid not in self.domains:
                continue
            dlist = self.domains[eachqid]
            llen = min(len(dlist), self.max_domains_num)
            for idy in range(llen):
                d_array[idx, idy]=dlist[idy]
                d_mask[idx, idy] = 1.0
        return d_array,d_mask

    def pad_sentences(self,sequences,wvmodel, dtype1='int32',dtype='float32'):
        '''
        @description: padding the sentences to the self.max_text_len.
        '''
        lengths = [min(len(s),self.max_text_len) for s in sequences]
        nb_samples = len(sequences)
        x = (np.zeros((nb_samples, self.max_text_len))).astype(dtype1)
        mask = (np.zeros((nb_samples,self. max_text_len))).astype(dtype)
        mask_last = (np.zeros((nb_samples, self.max_text_len))).astype(dtype)
        for idx, s in enumerate(sequences):
            mask_last[idx, lengths[idx] - 1] = 1.0
            for idy in range(lengths[idx]):
                wv = s[idy]
                mask[idx, idy] = 1.0
                if wv in wvmodel:
                    x[idx, idy] = wvmodel[wv]
                else:
                    x[idx, idy] = 0
        return x, mask,mask_last


    def banch_convert(self,batch, wvmodel,dtype1='float32',dtype2='int32'):
        '''
        @description: build the batch data according to the batch index list.
        @param batch{list}:batch data list 
        @param wvmodel{dict}:a dict which maps word to vector index num.
        @param dtype1{string}:the dtype of image matrix
        @param dtype2{string}:the dtype of domain
        '''
        qid_materials ={}
        posqid_materials = {}
        negqid_materials = {}
        batch_seq = batch.tolist()

        #qids
        qidlist = [x[0] for x in batch_seq]
        posqidlist = [x[1] for x in batch_seq]
        negqidlist = [x[2] for x in batch_seq]

        #sentences
        qid_sequences = [self.qid_fenci[x[0]].strip().split(' ') for x in batch_seq]
        posqid_sequences = [self.qid_fenci[x[1]].strip().split(' ') for x in batch_seq]
        negqid_sequences = [self.qid_fenci[x[2]].strip().split(' ') for x in batch_seq]
        #labels
        data_label = np.array([int(x[3]) for x in batch_seq])
        nb_samples = len(qidlist)
        domains_mat = (np.zeros((3,nb_samples, self.max_domains_num))).astype(dtype2)
        
        #images,domains,sentences of qids
        qid_materials['images'], qid_materials['images_mask'] = self.image_convert(qidlist)
        domains_mat[0, :, :],qid_materials['domains_mask'] = self.domains_convert(qidlist)
        qid_materials['sentences'],qid_materials['mask'],qid_materials['last_mask'] = self.pad_sentences(qid_sequences,wvmodel)

        #images,domains,sentences of positive qids
        posqid_materials['images'], posqid_materials['images_mask'] = self.image_convert(posqidlist)
        domains_mat[1, :, :],posqid_materials['domains_mask'] = self.domains_convert(posqidlist)
        posqid_materials['sentences'], posqid_materials['mask'],posqid_materials['last_mask'] = self.pad_sentences(posqid_sequences,wvmodel)

        #images,domains,sentences of negative qids
        negqid_materials['images'], negqid_materials['images_mask'] = self.image_convert(negqidlist)
        domains_mat[2, :, :],negqid_materials['domains_mask'] = self.domains_convert(negqidlist)
        negqid_materials['sentences'], negqid_materials['mask'],negqid_materials['last_mask'] = self.pad_sentences(negqid_sequences,wvmodel)
        return qid_materials,posqid_materials,negqid_materials,domains_mat,data_label

    def banch_convert_NoImages(self,batch, wvmodel, dtype1='float32',dtype2='int32'):
        '''
        @description: the simplified version of banch_convert, remove the image data.
        @param batch{list}:batch data list 
        @param wvmodel{dict}:a dict which maps word to vector index num.
        @param dtype1{string}:the dtype of image matrix
        @param dtype2{string}:the dtype of domain
        '''

        qid_materials ={}
        posqid_materials = {}
        negqid_materials = {}
        batch_seq = batch.tolist()

        qidlist = [x[0] for x in batch_seq]
        posqidlist = [x[1] for x in batch_seq]
        negqidlist = [x[2] for x in batch_seq]

        qid_sequences = [self.qid_fenci[x[0]].strip().split(' ') for x in batch_seq]
        posqid_sequences = [self.qid_fenci[x[1]].strip().split(' ') for x in batch_seq]
        negqid_sequences = [self.qid_fenci[x[2]].strip().split(' ') for x in batch_seq]

        data_label = np.array([int(x[3]) for x in batch_seq])
        
        nb_samples = len(qidlist)
        domains_mat = (np.zeros((3,nb_samples, self.max_domains_num))).astype(dtype2)
        domains_mat[0, :, :],qid_materials['domains_mask'] = self.domains_convert(qidlist)
        qid_materials['sentences'],qid_materials['mask'],qid_materials['last_mask'] = self.pad_sentences(qid_sequences,wvmodel)
        domains_mat[1, :, :],posqid_materials['domains_mask'] = self.domains_convert(posqidlist)
        posqid_materials['sentences'], posqid_materials['mask'],posqid_materials['last_mask'] = self.pad_sentences(posqid_sequences,wvmodel)
        domains_mat[2, :, :],negqid_materials['domains_mask'] = self.domains_convert(negqidlist)
        negqid_materials['sentences'], negqid_materials['mask'],negqid_materials['last_mask'] = self.pad_sentences(negqid_sequences,wvmodel)
        return qid_materials,posqid_materials,negqid_materials,domains_mat,data_label

    def next_batch(self,wvmodel):
        '''
        @description: fetch the data by batch
        @param wvmodel{dict}:a dict which maps word to vector index num.
        '''
        start_index = self.pos
        end_index = min(self.pos + self.batch_size, self.data_size)
        if self.pos + self.batch_size >= self.data_size:
            self.pos = self.data_size
            self.end = True
        else:
            self.pos += self.batch_size
        batch_data = self.data[start_index:end_index]
        qid_materials, posqid_materials, negqid_materials, domains_mat, data_label = self.banch_convert(batch = batch_data,wvmodel=wvmodel)
        return batch_data,qid_materials,posqid_materials,negqid_materials,domains_mat,data_label

    def next_batch_NoImages(self,wvmodel):
        '''
        @description: the simplified version of next_batch, remove the image data.
        @param wvmodel{dict}:a dict which maps word to vector index num.
        '''
        start_index = self.pos
        end_index = min(self.pos + self.batch_size, self.data_size)
        if self.pos + self.batch_size >= self.data_size:
            self.pos = self.data_size
            self.end = True
        else:
            self.pos += self.batch_size
        batch_data = self.data[start_index:end_index]

        qid_materials, posqid_materials, negqid_materials, domains_mat, data_label = self.banch_convert_NoImages(batch = batch_data,wvmodel=wvmodel)
        return batch_data,qid_materials,posqid_materials,negqid_materials,domains_mat,data_label


    def creat_images_batch(self):
        self.image_keys = list(self.images_mat.keys())
        self.im_sizes = len(self.image_keys)
        self.im_pos = 0
        self.im_end = False

    def images_batch_reset(self):
        self.im_pos = 0
        self.im_end = False


    def next_batch_images(self):
        start_index = self.im_pos
        end_index = min(self.im_pos + self.batch_size, self.im_sizes)
        if self.im_pos + self.batch_size >= self.im_sizes:
            self.im_pos = self.im_sizes
            self.im_end = True
        else:
            self.im_pos += self.batch_size
        batch_data = self.image_keys[start_index:end_index]
        blen = len(batch_data)
        x = (np.zeros((blen, self.image_height, self.image_width),dtype=np.float32))
        for i in range(blen):
            x[i,:,:]= self.images_mat[batch_data[i]]
        return x,blen



def load_word2vec_matrix(w2vpath, embedding_size):
    '''
    @description: read the pretrained word2vec model
    @param w2vpath{string}:the path of word2vec model
    @param embedding_size{int}:the size of word embedding vectors
    @return: vector(word embedding vectors),wvmodel(dict of word2index),vocab_size
    '''

    word2vec_file = w2vpath
    wvmodel ={}
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
        return vector,wvmodel,vocab_size
    else:
        logging.info("✘ The word2vec file doesn't exist. "
                     "Please use function <create_vocab_size(embedding_size)> to create it!")


if __name__ == '__main__':
    # Test Code
    filepath='../../data'
    embedding_size = 100

    word2vec_matrix, wvmodel, vocab_size = load_word2vec_matrix(
    os.path.join(filepath,'word2vec_' + str(embedding_size),'w2vmodel') ,embedding_size)

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
              "margin": 0.5,
              'l2_reg': 0.00004,
              'score_layer_size1': 200
              }
    data = Data(config, images={}, images_emb_map={}, is_train=True)
    data.load_domains(os.path.join(filepath, 'knowledge_num_list.txt'))
    # data.reload_train_data_with_num(trainpath=os.path.join(filepath, 'Train.json'), compare_neg_num=2)
    data.shuffle()
    batch_num = data.batch_num
    print(batch_num)
    while not data.end:
        _,qid_materials, posqid_materials, negqid_materials, domains_mat, data_label = data.next_batch_NoImages(wvmodel=wvmodel)
        print(qid_materials['domains_mask'].shape,
              posqid_materials['sentences'].shape, data_label.shape)




