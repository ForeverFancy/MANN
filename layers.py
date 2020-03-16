import torch
import torch.nn as nn


class SentenceEmbedding(nn.Module):
    '''
    SentenceEmbedding for Multimodal Exercise Representing Layer, convert words in input text sequence to embeddings.
    '''

    def __init__(self, word2vec_matrix):
        super(SentenceEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(torch.tensor(word2vec_matrix))

    def forward(self, x) -> torch.tensor:
        '''
        @param x of shape (batch_size, sequence_length): input sequence of word index

        @return out of shape (batch_size, sequence_length, embedding_dim): output after embedding
        '''
        out = self.embedding_layer(x)
        return out


class DomainEmbedding(nn.Module):
    '''
    Domain embedding in Multimodal Exercise Representing layer.
    '''

    def __init__(self, all_domains_num: int, max_domains_num: int, domain_embedding_dim: int):
        '''
        @param all_domains_num: all domain nums (L_all in paper)

        @param max_domains_num: max domain that could be contained in one question

        @param domain_embedding_dim: embedding dimension of domain (d_2 in paper)
        '''
        super(DomainEmbedding, self).__init__()
        self.domain_emb = nn.Embedding(
            num_embeddings=all_domains_num, embedding_dim=domain_embedding_dim)
        self.max_domain_nums = max_domains_num

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        @param x of shape (batch_size, max_domains_num): each one is a domain id list

        @return out of shape (batch_size, max_domains_num, domain_embedding_dim)
        '''

        out = self.domain_emb(x)
        return out


class AttentionLSTM(nn.Module):
    '''
    Attention-based LSTM in Multimodal Exercise Representing Layer.
    '''

    def __init__(self, input_size: int, hidden_size: int):
        '''
        @param input_size: input dimension, word embedding size + domain embedding size ( + image_embedding size)

        @param hidden_size: hidden state size
        '''
        super(AttentionLSTM, self).__init__()
        self.LSTM_cell = nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, bias=True)

    def forward(self, x: torch.tensor, last_hidden_state: tuple) -> torch.tensor:
        '''
        @param x of shape(batch_size, input_size)

        @param last_hidden_state (tuple(tensor, tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size. First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.

        @return hidden_state contains (hidden_hidden of shape (batch_size, hidden_size), hidden_cell)
        '''
        hidden_state = self.LSTM_cell.forward(x, last_hidden_state)
        return hidden_state


class MER(nn.Module):
    '''
    Multimodal Exercise Representing Layer.
    '''

    def __init__(self, word2vec_matrix, word_embedding_dim: int, all_domains_num: int, max_domains_num: int, domain_embedding_dim: int, hidden_size: int, device: torch.device, attention_size: int = 100):
        super(MER, self).__init__()
        self.sentence_emb = SentenceEmbedding(word2vec_matrix)
        self.max_domains_num = max_domains_num
        self.hidden_size = hidden_size
        self.domain_emb = DomainEmbedding(
            all_domains_num, max_domains_num, domain_embedding_dim)
        self.attention_LSTM = AttentionLSTM(
            input_size=domain_embedding_dim + word_embedding_dim, hidden_size=hidden_size)
        self.W_ac = nn.Parameter(torch.zeros(
            (word_embedding_dim + domain_embedding_dim + hidden_size, attention_size)))
        self.V_ac = nn.Parameter(torch.zeros((attention_size, 1)))
        nn.init.xavier_uniform_(self.W_ac)
        nn.init.xavier_uniform_(self.V_ac)
        self.device = device

    def forward(self, text, domains, domains_mask, sentence_last_mask) -> torch.tensor:
        '''
        @param text of shape (batch_size, sentence_length): input text

        @param domains of shape (batch_size, max_domains_num): input question's domains

        @param domains_mask of shape (batch_size, max_domains_num): mask of domains (mask of src or padding)

        @param sentence_last_mask of shape (batch_size, sentence_length): mask of the sentence's last position without padding

        @return hidden_states of shape (batch_size, sentence_length, hidden_size): hidden_states matrix, h^E in paper

        @return last_hidden_states of shape (batch_size, hidden_size): last hidden state, r^E in paper
        '''
    
        text_embedding = self.sentence_emb(text).to(dtype=torch.float)
        domains_embedding = self.domain_emb(domains)
        hidden_state = None
        sentence_length = text.size(1)
        hidden_states = []                  # h in paper
        last_hidden_state = None            # r in paper

        for i in range(sentence_length):
            word = text_embedding[:, i, :]
            hidden_state = self.step(word=word, last_hidden_state=hidden_state,
                                     domains_embedding=domains_embedding, domains_mask=domains_mask)
            hidden_states.append(hidden_state[0])

        hidden_states = torch.cat(
            hidden_states, dim=1).reshape(-1, sentence_length, self.hidden_size)

        last_hidden_state = torch.bmm(
            sentence_last_mask.unsqueeze(1), hidden_states).squeeze(1)

        return hidden_states, last_hidden_state

    def step(self, word: torch.tensor, last_hidden_state: tuple, domains_embedding: torch.tensor, domains_mask: torch.tensor):
        '''
        Perform a step of LSTM Cell.

        @param word of shape (batch_size, word_embedding_dim): w_t in paper

        @param last_hidden_state, tuple of shape ((batch_size, hidden_size), (batch_size, hidden_size)): hidden_state and cell at time t-1, h_{t-1} and c_{t-1} in paper

        @param domains_embedding of shape (batch_size, max_domains_num, domain_embedding_dim): input question domains embedding, u in paper

        @param domains_mask of shape (batch_size, max_domains_num): mask of domains (mask of src or padding)

        @return hidden_state contains (hidden_hidden of shape (batch_size, hidden_size), hidden_cell)
        '''
        
        batch_size = word.size(0)
        word = torch.unsqueeze(word, dim=1)
        expand_word = word.expand(batch_size, self.max_domains_num, word.size(2))
        if last_hidden_state is None:
            expanded_hidden_state = torch.zeros(
                (batch_size, self.max_domains_num, self.hidden_size), dtype=torch.float).to(self.device)
        else:
            expanded_hidden_state = torch.unsqueeze(last_hidden_state[0], dim=1).expand(
            last_hidden_state[0].size(0), self.max_domains_num, last_hidden_state[0].size(1))
        attention_mat = torch.cat(
            [domains_embedding, expand_word, expanded_hidden_state], dim=2)

        w_proj = torch.matmul(attention_mat, self.W_ac)
        v_proj = torch.matmul(torch.tanh(w_proj), self.V_ac).squeeze(2)

        v_proj.data.masked_fill_(domains_mask.bool(), float('-inf'))

        # shape (batch_size, max_domains_num)
        alpha = nn.functional.softmax(v_proj, dim=1)

        u_t = torch.bmm(alpha.unsqueeze(1), domains_embedding).squeeze(1)  # shape (batch_size, domain_embedding_dim)
        x = torch.cat([word.squeeze(1), u_t], dim=1)

        hidden_state = self.attention_LSTM.forward(x, last_hidden_state)

        return hidden_state


class SimilarityAttention(nn.Module):
    def __init__(self):
        super(SimilarityAttention, self).__init__()

    def forward(self, hidden_states_a: torch.tensor, hidden_states_b: torch.tensor, last_hidden_state_a: torch.tensor, last_hidden_state_b: torch.tensor, sentence_mask_a: torch.tensor, sentence_mask_b: torch.tensor) -> torch.tensor:
        '''
        @param hidden_states_a of shape (batch_size, sentence_length, hidden_size)

        @param hidden_states_b of shape (batch_size, sentence_length, hidden_size)

        @param last_hidden_state_a of shape (batch_size, hidden_size)

        @param last_hidden_state_b of shape (batch_size, hidden_size)

        @param sentence_mask_a of shape (batch_size, sentence_length)

        @param sentence_mask_b of shape (batch_size, sentence_length)

        @return h_att_a of shape (batch_size, hidden_size)

        @return h_att_b of shape (batch_size, hidden_size)

        @return s_a of shape (batch_size, sentence_length)

        @return s_b of shape (batch_size, sentence_length)
        '''
        self.build_attention_matrix(hidden_states_a, hidden_states_b, sentence_mask_a, sentence_mask_b)
        # attention_matrix, shape (batch_size, sentence_length, sentence_length)
        s_a = torch.sum(self.attention_matrix, dim=2)
        s_b = torch.sum(self.attention_matrix, dim=1)

        h_att_a = self.build_attention_hidden_state(last_hidden_state_b, hidden_states_a, sentence_mask_a)
        h_att_b = self.build_attention_hidden_state(last_hidden_state_a, hidden_states_b, sentence_mask_b)

        return s_a, s_b, h_att_a, h_att_b

    def build_attention_matrix(self, hidden_states_a: torch.tensor, hidden_states_b: torch.tensor, sentence_mask_a: torch.tensor, sentence_mask_b: torch.tensor):
        '''
        @param hidden_states_a of shape (batch_size, sentence_length, hidden_size)

        @param hidden_states_b of shape (batch_size, sentence_length, hidden_size)

        @param sentence_mask_a of shape (batch_size, sentence_length)

        @param sentence_mask_b of shape (batch_size, sentence_length)

        '''
        batch_size = hidden_states_a.size(0)
        sentence_length = hidden_states_a.size(1)
        self.attention_matrix = torch.zeros((batch_size, sentence_length, sentence_length))
        sentence_mask_a = sentence_mask_a.unsqueeze(2)
        sentence_mask_b = sentence_mask_b.unsqueeze(2)

        # Element wise multiply, add small num prevent overflow (div 0)
        mask_hidden_states_a = hidden_states_a * sentence_mask_a + 1e-6
        mask_hidden_states_b = hidden_states_b * sentence_mask_b + 1e-6

        norm_a = torch.sum(torch.pow(mask_hidden_states_a, 2), dim=2).sqrt().unsqueeze(2)            # shape (batch_size, sentence_length, 1)
        norm_b = torch.sum(torch.pow(mask_hidden_states_b, 2), dim=2).sqrt().unsqueeze(2)

        normalized_a = mask_hidden_states_a / norm_a
        normalized_b = mask_hidden_states_b / norm_b

        self.attention_matrix = torch.bmm(normalized_a, normalized_b.transpose(1, 2))
        
    def build_attention_hidden_state(self, last_hidden_state: torch.tensor, target_hidden_states: torch.tensor, target_sentence_mask: torch.tensor) -> torch.tensor:
        '''
        @param last_hidden_state of shape (batch_size, hidden_size)

        @param target_hidden_states of shape (batch_size, sentence_length, hidden_size)

        @param target_sentence_mask of shape (batch_size, sentence_length)

        @return attention_hidden_state of shape (batch_size, hidden_size)
        '''
        batch_size = target_hidden_states.size(0)
        sentence_length = target_hidden_states.size(1)
        sentence_mask = target_sentence_mask.unsqueeze(2)
        
        # shape (batch_size, sentence_length, hidden_size)
        mask_hidden_states = target_hidden_states * sentence_mask + 1e-6        
        
        # shape (batch_size, sentence_length, 1)
        norm = torch.sum(torch.pow(mask_hidden_states, 2), dim=2).sqrt().unsqueeze(2)            

        normalized = mask_hidden_states / norm
        
        # (batch_size, 1, hidden_size) * (batch_size, hidden_size, sentence_length) -> (batch_size, 1, sentence_length)
        alpha = torch.bmm(last_hidden_state.unsqueeze(1), normalized.transpose(1, 2))  
        
        # (batch_size, 1, sentence_length) * (batch_size, sentence_length, hidden_size) -> (batch_size, 1, hidden_size)
        attention_hidden_state = torch.bmm(alpha, mask_hidden_states).squeeze(1)
        return attention_hidden_state


class SimilarityScore(nn.Module):
    def __init__(self, input_feature_size: int, proj_size: int, dropout_rate: float = 0.2):
        super(SimilarityScore, self).__init__()
        self.att_proj = nn.Linear(
            in_features=input_feature_size, out_features=proj_size, bias=True)
        self.score_proj = nn.Linear(
            in_features=proj_size, out_features=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.att_proj(x)
        out = torch.relu(out)
        out = self.dropout(out)
        score = torch.sigmoid(self.score_proj(out))

        return score
