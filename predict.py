import torch
from tqdm import tqdm,trange
from basicmodel import Mymodel
from model import BiLSTM_CRF
import random
from dataloader import get_data,loadData,prepare_sequence
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--DATA_PATH', type=str, default='data/')
parser.add_argument('--CPT_PATH', type=str, default='use_checkpoint/')
parser.add_argument('--SAVE_PATH', type=str, default='answer/')
parser.add_argument('--DEV_BATCH_SIZE', type=int, default=1)
parser.add_argument('--GPU_DEVICE', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_DEVICE
DATA_PATH = args.DATA_PATH
CPT_PATH = args.CPT_PATH
DEV_BATCH_SIZE = args.DEV_BATCH_SIZE
SAVE_PATH = args.SAVE_PATH
D1 = 0.65
D2 = 0.9
random.seed(1234)
def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
def build_corpus(textPath, tagPath, make_vocab=True, data_dir="data"):
    """读取数据"""
    word_lists = []
    with open(textPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word_lists.append(line.split())
    tag_lists = []
    with open(tagPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tag_lists.append(line.split())

    # dataset = list(zip(word_lists, tag_lists))

    word_lists = sorted(word_lists, key=lambda x: len(x), reverse=True)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=True)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        word2id['<UNK>'] = len(word2id)
        word2id['<PAD>'] = len(word2id)

        tag2id['<PAD>'] = len(tag2id)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

torch.manual_seed(1)


def idx_to_tagList(idxList, ix_to_tag):
    tagList = []
    for idxl in idxList:
        temp = []
        for idx in idxl:
            temp.append(ix_to_tag[idx])
        tagList.append(temp)
    return tagList


loader, word_to_ix, tag_to_ix, word_set,dev_dataloader = get_data(DATA_PATH, 32, 1)
ix_to_tag = [i for i in tag_to_ix]
test_set, _, _ = loadData("data/test.txt", "data/dev_TAG.txt")
test_model = torch.load(CPT_PATH + "checkpoint_1400.pkl")

test_set = test_set
with open(SAVE_PATH + "1.txt", "w", encoding="utf-8") as f:
    for sentence in tqdm(test_set):
        precheck_sent = prepare_sequence(sentence, word_to_ix).unsqueeze(0)
        result = idx_to_tagList(test_model.predict(precheck_sent.cuda()), ix_to_tag)
        for res in result:
            f.write(' '.join(res))
            f.write('\n')
test_model = torch.load(CPT_PATH + "checkpoint_1900.pkl")
with open(SAVE_PATH + "2.txt", "w", encoding="utf-8") as f:
    for sentence in tqdm(test_set):
        precheck_sent = prepare_sequence(sentence, word_to_ix).unsqueeze(0)
        result = idx_to_tagList(test_model.predict(precheck_sent.cuda()), ix_to_tag)
        for res in result:
            f.write(' '.join(res))
            f.write('\n')


device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_data, train_tag, word_2_index, tag_2_index = build_corpus("./data/train.txt", "./data/train_TAG.txt")
index_2_tag = [i for i in tag_2_index]
model = torch.load(CPT_PATH + "checkpoint_23.pkl")

word_lists = []
fv = open(SAVE_PATH + "3.txt", "w", encoding='utf-8')
with open("data/test.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        word_lists.append(line.split())
for i in range(len(word_lists)):
    for j in range(len(word_lists[i])):
        if word_lists[i][j] not in word_2_index.keys():
            word_lists[i][j] = "<UNK>"
for i in trange(len(word_lists)):
    text_index = [[word_2_index[word] for word in word_lists[i]]]
    text_index = torch.tensor(text_index, dtype=torch.int64, device=device)
    model.forward(text_index)
    pre = [index_2_tag[i] for i in model.pre]
    s = ' '.join(pre) + '\n'
    fv.write(s)

dev1 = []
with open(SAVE_PATH + '1.txt', 'r') as f:
    for line in f:
        dev1.append(line)

dev2 = []
with open(SAVE_PATH + '2.txt', 'r') as f:
    for line in f:
        dev2.append(line)

dev3 = []
with open(SAVE_PATH + '3.txt', 'r') as f:
    for line in f:
        dev3.append(line)

bagging = []
for ans1, ans2, ans3 in zip(dev1, dev2, dev3):
    if random.random() > D2:
        if random.random() > D1:
            bagging.append(ans1)
        else:
            bagging.append(ans2)
    else:
        bagging.append(ans3)
with open(SAVE_PATH + 'answer.txt', 'w', encoding='utf-8') as f:
    for line in bagging:
        f.write(line)
print("predict finish!!!")