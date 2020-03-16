import torch
import torch.nn as nn
from Data_Process import *
from model import *


def Rank_precision_recall_at(score_label_sorted, all_posnum, topk):
    '''
    @description: compute the precision, recall and f1 score with specified top-k.
    @param score_label_sorted{dict}:{'qid':[(pred_score1,label1),(pred_score2,label2),...]} 
    @param all_posnum{dict}:{'qid':the num of its positive example}
    @param topk{int}: specified top-k 
    @return: res_pr, res_rc, F1_value, qid_result
    '''
    pr_score = 0.0
    recall_score = 0.0
    qidnum = 0
    F1_score = 0.0
    qid_result = {}
    for qid, score_labal in score_label_sorted.items():
        qidnum += 1
        len1 = min(len(score_labal), topk)
        hit = 0
        for i in range(len1):
            if score_labal[i][1] == 1:
                hit += 1
        qid_pr = float(hit)/float(topk)
        pr_score += qid_pr
        qid_recall = float(hit)/float(all_posnum[qid])
        recall_score += qid_recall
        qid_F1 = 2 * float(hit) / (float(topk) + float(all_posnum[qid]))
        F1_score += qid_F1
        qid_result[qid] = [qid_pr, qid_recall, qid_F1]

    res_pr = pr_score / qidnum
    res_rc = recall_score / qidnum
    F1_value = F1_score / qidnum
    return res_pr, res_rc, F1_value, qid_result


def train(epoch: int, device: torch.device):
    filepath = '../data'
    output = '../result'
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
              "max_text_len": 200,
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

    print('***********load traindata**********')
    traindata = Data(config=config, images=images,
                     images_emb_map=images_emb_map, is_train=True)
    traindata.load_domains(os.path.join(
        filepath, 'knowledge_num_list.txt'))

    print('***********load testdata***********')
    testdata = Data(config=config, images=images,
                    images_emb_map=images_emb_map, is_train=False)
    testdata.load_domains(os.path.join(filepath, 'knowledge_num_list.txt'))

    model = MANN(word2vec_matrix,
                 word_embedding_dim=embedding_size,
                 all_domains_num=config['all_domains_num'],
                 max_domains_num=config['max_domains_num'],
                 domain_embedding_dim=domain_emb_size,
                 hidden_size=domain_emb_size + embedding_size,
                 device=device,
                 max_sentence_length=config['max_text_len'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.0000004)
    model.to(device)
    model.train()
    ploss = PairwiseLoss(margin=config["margin"], device=device)

    for i in range(epoch):
        traindata.reload_train_data_with_num(
            trainpath=os.path.join(filepath, 'Train.json'), compare_neg_num=2)
        batch_num = traindata.batch_num
        traindata.shuffle()
        model.train()
        model.zero_grad()
        total_loss = 0.0

        batch_no = 0
        while not traindata.end:
            batch_data, qid_materials, posqid_materials, negqid_materials, domains_mat, data_label = traindata.next_batch_NoImages(wvmodel=wvmodel)
            
            score_positive, score_negitive = model.forward(qid_materials, posqid_materials,
                            negqid_materials, domains_mat)
            print(score_positive, score_negitive)
            loss = ploss(score_positive, score_negitive)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(("Epoch{0}-batch({1}/{2}): loss={3}".format(i,batch_no,batch_num,loss)))
            batch_no += 1

        print(("Epoch-{0}: overall_loss={1}".format(i, total_loss)))
        print('=' * 50)
        
        if i >= 0:
            testdata.reset()
            pred_label = []

            while not testdata.end:
                batch_data, qid_materials, posqid_materials, negqid_materials, domains_mat, data_label = testdata.next_batch_NoImages(wvmodel=wvmodel)
                model.eval()
                with torch.no_grad():
                    # only consider score, because posid and negid are set the same when test
                    pred_all, _ = model.forward(qid_materials, posqid_materials, negqid_materials, domains_mat)
                
                batch1 = batch_data.tolist()
                batchnum = len(batch1)
                for idx in range(batchnum):
                    pred_label.append((batch1[idx][0], batch1[idx][1], batch1[idx][2], pred_all[idx], data_label[idx]))
            
            qid_score_label_tmp = {}
            qid_all_posnum = {}
            for eachpred in pred_label:
                if eachpred[0] not in qid_score_label_tmp:
                    qid_score_label_tmp[eachpred[0]] = []

                qid_score_label_tmp[eachpred[0]].append((eachpred[3], eachpred[4]))

                if eachpred[0] not in qid_all_posnum:
                    qid_all_posnum[eachpred[0]] = 0
                if eachpred[4] == 1:
                    qid_all_posnum[eachpred[0]] += 1

            qid_score_label = {}
            for qid, score_labal in qid_score_label_tmp.items():
                qid_score_label[qid] = sorted(score_labal, key=lambda asd: asd[0], reverse=True)
            
            topn = 1
            while topn <= 10:
                precision_at, recall_at, F1_value_at, _ = Rank_precision_recall_at(
                        qid_score_label, qid_all_posnum, topn)
                # print(("epoch = {0}, top n ={1},precision_at={2}, recall_at={3}, F1_value_at={4}".format(
                #         i, topn, precision_at, recall_at, F1_value_at)))
                sys.stdout.flush()
                with open(os.path.join(output, 'test_result.txt'), 'a+') as outfile:
                    outfile.write(("epoch = {0},top n ={1}, precision_at={2}, recall_at={3}, F1_value_at={4}\n".format(
                            i, topn, precision_at, recall_at, F1_value_at)))
                topn = topn + 1
            with open(os.path.join(output, 'test_result.txt'), 'a+') as outfile:
                outfile.write('=' * 50)
                

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(10, device)
