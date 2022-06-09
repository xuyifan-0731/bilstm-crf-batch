import torch
import torch.utils.data as Data
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader



class MyDataset(Dataset):
    def __init__(self, datas, tags, word_2_index, tag_2_index):  # 把所有数据保存下来
        self.datas = datas
        self.tags = tags
        self.word_2_index = word_2_index
        self.tag_2_index = tag_2_index

    def __getitem__(self, index):  # 获取每一条数据 index是行索引
        data = self.datas[index]
        tag = self.tags[index]
        data_index = [self.word_2_index.get(i, self.word_2_index["<UNK>"]) for i in data]
        tag_index = [self.tag_2_index[i] for i in tag]
        return data_index, tag_index

    def __len__(self):  # 返回总行数
        assert len(self.datas) == len(self.tags)
        return len(self.tags)

    def pro_batch_data(self, batch_datas):  # 对每个batch进行填充
        datas = []
        tags = []
        batch_lens = []
        for data, tag in batch_datas:
            datas.append(data)
            tags.append(tag)
            batch_lens.append(len(data))
        batch_max_len = max(batch_lens)

        datas = [i + [self.word_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in datas]
        tags = [i + [self.tag_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in tags]

        return torch.tensor(datas, dtype=torch.int64), torch.tensor(tags, dtype=torch.long)


def build_tag2id(lists):
    maps = {}
    lists.append(["<START>"])
    lists.append(["<STOP>"])
    lists.append(["[PAD]"])
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    lists.pop()
    lists.pop()
    lists.pop()
    return maps


def build_word2id(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps


def loadData(textPath, tagPath):
    sents = []
    with open(textPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sents.append(line.split())
    tags = []
    with open(tagPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tags.append(line.split())

    dataset = list(zip(sents, tags))
    return sents, tags, dataset


def get_dict(word_lists, tag_lists):
    word2id = build_word2id(word_lists)
    tag2id = build_tag2id(tag_lists)
    word2id['<UNK>'] = len(word2id)
    word2id['<PAD>'] = len(word2id)
    tag2id['<PAD>'] = len(tag2id)
    return word2id, tag2id


def to_tag(tag_list, ix_to_tag):
    temp = []
    for tag in tag_list:
        temp.append(ix_to_tag[tag])

    return temp


def predict(model, sentence_set, word_to_ix, ix_to_tag, pre_tag_path=None):
    if pre_tag_path == None:
        pre_tags = []
        for sentence in tqdm(sentence_set):
            precheck_sent = prepare_sequence(sentence, word_to_ix).cuda()
            score, tags = model(precheck_sent)
            pre_tags.extend(to_tag(tags, ix_to_tag))

        return pre_tags
    else:
        with open(pre_tag_path, "w") as f:
            for sentence in tqdm(sentence_set):
                precheck_sent = prepare_sequence(sentence, word_to_ix)
                score, tags = model(precheck_sent)
                f.write(' '.join(to_tag(tags, ix_to_tag)))
                f.write('\n')


def trunc_pad(_list: list, max_len=100, if_sentence=True):  # 设置输入句子的最大长度
    if len(_list) == max_len:
        return _list
    if len(_list) > max_len:
        if if_sentence:
            return ["[PAD]"] * 100
        else:

            return ["O"] * 100
    else:
        if if_sentence:
            _list.extend(["[PAD]"] * (max_len - len(_list)))
            return _list
        else:
            _list.extend(["O"] * (max_len - len(_list)))
            return _list


def prepare_sequence(seq, to_ix):
    # idxs = [to_ix[w] for w in seq]
    idxs = []
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(to_ix['<UNK>'])
    return torch.tensor(idxs, dtype=torch.long)


def idx_to_tagList(idxList, ix_to_tag):
    tagList = []
    for idxl in idxList:
        temp = []
        for idx in idxl:
            temp.append(ix_to_tag[idx])
        tagList.append(temp)
    return tagList


def get_data(DATA_PATH, BATCH_SIZE, DEV_BATCH_SIZE):
    train_sents, train_tags, train_datas = loadData(DATA_PATH + "train.txt", DATA_PATH + "train_TAG.txt")
    # train_sents = train_sents[:TRAIN_SIZE]
    # train_tags = train_tags[:TRAIN_SIZE]

    word_to_ix, tag_to_ix = get_dict(train_sents, train_tags)
    dev_sents, dev_tags, dev_data = loadData(DATA_PATH + "dev.txt", DATA_PATH + "dev_TAG.txt")
    ix_to_tag = [i for i in tag_to_ix]

    dev_dataset = MyDataset(dev_sents, dev_tags, word_to_ix, tag_to_ix)
    dev_dataloader = DataLoader(dev_dataset, DEV_BATCH_SIZE, shuffle=False,
                                collate_fn=dev_dataset.pro_batch_data)  # 已经按长度排序，不能shuffle

    word_set = train_sents
    label_set = train_tags
    for item_st in range(len(word_set)):
        word_set[item_st] = trunc_pad(word_set[item_st], if_sentence=True)
        label_set[item_st] = trunc_pad(label_set[item_st], if_sentence=False)

    for sentence in word_set:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    for i in range(len(word_set)):
        word_set[i] = [word_to_ix[t] for t in word_set[i]]
        label_set[i] = [tag_to_ix[t] for t in label_set[i]]

    # 先转换成 torch 能识别的 Dataset
    torch_dataset = Data.TensorDataset(torch.tensor(word_set, dtype=torch.long),
                                       torch.tensor(label_set, dtype=torch.long))

    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  #
    )

    return loader, word_to_ix, tag_to_ix, word_set
