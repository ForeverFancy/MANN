import torch
import torch.nn
from layers import *
import unittest


class MANN(nn.Module):
    '''
    Multimodal Attention-based Neural Network model.
    '''

    def __init__(self, word2vec_matrix, word_embedding_dim: int, all_domains_num: int, max_domains_num: int, domain_embedding_dim: int, hidden_size: int, device: torch.device, attention_size: int = 100, max_sentence_length: int = 100):
        super(MANN, self).__init__()
        self.mer_layer = MER(word2vec_matrix, word_embedding_dim=word_embedding_dim, all_domains_num=all_domains_num,
                             max_domains_num=max_domains_num, domain_embedding_dim=domain_embedding_dim, hidden_size=hidden_size, device=device, attention_size=attention_size).to(device)
        self.sim_attention_layer = SimilarityAttention().to(device)
        self.sim_score_layer = SimilarityScore(
            input_feature_size=4 * hidden_size + 2 * max_sentence_length, proj_size=200).to(device)
        self.device = device

    def forward(self, qid_materials, posqid_materials, negqid_materials, domains_mat):
        qid_input = torch.tensor(qid_materials['sentences'], dtype=torch.long).to(self.device)
        qid_mask = torch.tensor(qid_materials['mask']).to(self.device)
        qid_mask_last = torch.tensor(qid_materials['last_mask']).to(self.device)
        posqid_input = torch.tensor(posqid_materials['sentences'], dtype=torch.long).to(self.device)
        posqid_mask = torch.tensor(posqid_materials['mask']).to(self.device)
        posqid_mask_last = torch.tensor(posqid_materials['last_mask']).to(self.device)
        negqid_input = torch.tensor(negqid_materials['sentences'], dtype=torch.long).to(self.device)
        negqid_mask = torch.tensor(negqid_materials['mask']).to(self.device)
        negqid_mask_last = torch.tensor(negqid_materials['last_mask']).to(self.device)
        # self.qid_images: qid_materials['images']
        # self.posqid_images: posqid_materials['images']
        # self.negqid_images: negqid_materials['images']
        # self.images :images_mat
        qid_domains = torch.tensor(domains_mat[0, :, :], dtype=torch.long).to(self.device)
        posqid_domains = torch.tensor(domains_mat[0, :, :], dtype=torch.long).to(self.device)
        negqid_domains = torch.tensor(domains_mat[0, :, :], dtype=torch.long).to(self.device)
        # self.qid_images_mask:qid_materials['images_mask']
        qid_domains_mask = torch.tensor(qid_materials['domains_mask']).to(self.device)
        # self.posqid_images_mask: posqid_materials['images_mask']
        posqid_domains_mask = torch.tensor(posqid_materials['domains_mask']).to(self.device)
        # self.negqid_images_mask: negqid_materials['images_mask']
        negqid_domains_mask = torch.tensor(negqid_materials['domains_mask']).to(self.device)

        qid_hidden_states, qid_last_hidden_state = self.mer_layer.forward(
            qid_input, qid_domains, qid_domains_mask, qid_mask_last)
        posqid_hidden_states, posqid_last_hidden_state = self.mer_layer.forward(
            posqid_input, posqid_domains, posqid_domains_mask, posqid_mask_last)
        negqid_hidden_states, negqid_last_hidden_state = self.mer_layer.forward(
            negqid_input, negqid_domains, negqid_domains_mask, negqid_mask_last)

        h_att_q1, h_att_p, s_q1, s_p = self.sim_attention_layer.forward(qid_hidden_states, posqid_hidden_states,
                                         qid_last_hidden_state, posqid_last_hidden_state, qid_mask, posqid_mask)
        h_att_q2, h_att_n, s_q2, s_n = self.sim_attention_layer.forward(qid_hidden_states, negqid_hidden_states,
                                         qid_last_hidden_state, negqid_last_hidden_state, qid_mask, negqid_mask)

        score_positive = self.sim_score_layer.forward(torch.cat([qid_last_hidden_state, h_att_q1, s_q1, s_p, h_att_p, posqid_last_hidden_state], dim=1))
        score_negitive = self.sim_score_layer.forward(torch.cat([qid_last_hidden_state, h_att_q2, s_q2, s_n, h_att_n, negqid_last_hidden_state], dim=1))
        # print(score_positive)

        return score_positive, score_negitive


class PairwiseLoss(nn.Module):
    '''
    Pairwise loss function in paper.
    '''
    def __init__(self, margin: float, device: torch.device):
        '''
        @param margin: µ in paper, forcing S(E, E_{s}) to be greater than S(E, E_{ds}) by µ
        '''
        super(PairwiseLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, score_positive: torch.tensor, score_negitive: torch.tensor):
        loss = torch.sum(torch.max(torch.zeros(score_positive.shape).to(self.device), self.margin - (score_positive - score_negitive)))
        return loss
