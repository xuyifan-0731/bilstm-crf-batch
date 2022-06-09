import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_bacth(vec):
    max_score_vec = torch.max(vec, dim=1)[0]
    max_score_broadcast = max_score_vec.view(vec.shape[0], -1).expand(vec.shape[0], vec.size()[1])
    return max_score_vec + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self, bacth=1):
        return (torch.randn(2, bacth, self.hidden_dim // 2).cuda(),
                torch.randn(2, bacth, self.hidden_dim // 2).cuda())

    def _forward_alg(self, batchfeats):
        alpha_list = []
        for feats in batchfeats:
            # Do the forward algorithm to compute the partition function
            init_alphas = torch.full((1, self.tagset_size), -10000.)
            # START_TAG has all of the score.
            init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

            # Wrap in a variable so that we will get automatic backprop
            forward_var = init_alphas

            # Iterate through the sentence
            for feat in feats:
                alphas_t = []  # The forward tensors at this timestep
                for next_tag in range(self.tagset_size):
                    # broadcast the emission score: it is the same regardless of
                    # the previous tag
                    emit_score = feat[next_tag].view(
                        1, -1).expand(1, self.tagset_size)
                    # the ith entry of trans_score is the score of transitioning to
                    # next_tag from i
                    trans_score = self.transitions[next_tag].view(1, -1)
                    # The ith entry of next_tag_var is the value for the
                    # edge (i -> next_tag) before we do log-sum-exp
                    next_tag_var = forward_var + trans_score + emit_score
                    # The forward variable for this tag is log-sum-exp of all the
                    # scores.
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            alpha = log_sum_exp(terminal_var)
            alpha_list.append(alpha.view(1))
        return torch.cat(alpha_list)

    def _forward_alg_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        # if IF_CUDA:
        # init_alphas = torch.full((feats.shape[0], self.tagset_size), -10000.).cuda()
        # else:
        init_alphas = torch.full((feats.shape[0], self.tagset_size), -10000.).cuda()
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas  # [1,6]
        convert_feats = feats.permute(1, 0, 2)
        # Iterate through the sentence
        for feat in convert_feats:  # feat 6
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_tag].view(
                    feats.shape[0], -1).expand(feats.shape[0], self.tagset_size)  # [1,6]
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).repeat(feats.shape[0], 1)  # [1,6]
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score  # [1,6]
                # The forward variable for this tag is log-sum-exp of all the

                alphas_t.append(log_sum_exp_bacth(next_tag_var))

            forward_var = torch.stack(alphas_t).permute(1, 0)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].view(1, -1).repeat(feats.shape[0], 1)
        alpha = log_sum_exp_bacth(terminal_var)

        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden(bacth=len(sentence))
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # 计算给定tag序列的分数，即一条路径的分数
        totalsocre_list = []
        for feat, tag in zip(feats, tags):
            totalscore = torch.zeros(1).cuda()
            tag = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).cuda(), tag])
            for i, smallfeat in enumerate(feat):
                # 递推计算路径分数：转移分数 + 发射分数
                totalscore = totalscore + \
                             self.transitions[tag[i + 1], tag[i]] + smallfeat[tag[i + 1]]
            totalscore = totalscore + self.transitions[self.tag_to_ix[STOP_TAG], tag[-1]]
            totalsocre_list.append(totalscore)
        return torch.cat(totalsocre_list)

    def _viterbi_decode(self, feats):
        backpointers = []

        # 初始化viterbi的previous变量
        init_vvars = torch.full((1, self.tagset_size), -10000.).cpu()
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        previous = init_vvars
        for obs in feats:
            # 保存当前时间步的回溯指针
            bptrs_t = []
            # 保存当前时间步的viterbi变量
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                # 维特比算法记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的转移分数
                # 并不取决与当前tag的发射分数
                next_tag_var = previous.cpu() + self.transitions[next_tag].cpu()
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            # 更新previous，加上当前tag的发射分数obs
            previous = (torch.cat(viterbivars_t).cpu() + obs.cpu()).view(1, -1)
            # 回溯指针记录当前时间步各个tag来源前一步的tag
            backpointers.append(bptrs_t)

        # 考虑转移到STOP_TAG的转移分数
        terminal_var = previous.cpu() + self.transitions[self.tag_to_ix[STOP_TAG]].cpu()
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 通过回溯指针解码出最优路径
        best_path = [best_tag_id]
        # best_tag_id作为线头，反向遍历backpointers找到最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # 去除START_TAG
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _viterbi_decode_predict(self, feats_list):
        path_list = []
        for feats in feats_list:
            backpointers = []

            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.).cpu()
            init_vvars[0][self.tag_to_ix[START_TAG]] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = init_vvars
            for feat in feats:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    next_tag_var = forward_var.cpu() + self.transitions[next_tag].cpu()
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t).cpu() + feat.cpu()).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var.cpu() + self.transitions[self.tag_to_ix[STOP_TAG]].cpu()
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path.reverse()
            path_list.append(best_path)
        return path_list

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg_parallel(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def predict(self, sentence):
        # logger.info("\n lstm begin" + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        lstm_feats = self._get_lstm_features(sentence)
        # logger.info("\n vit begin" + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        # Find the best path, given the features.
        tag_seq_list = self._viterbi_decode_predict(lstm_feats)
        # logger.info("\n vit end" + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        return tag_seq_list
