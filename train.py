import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import logging.handlers
from tqdm import tqdm
from tqdm import trange
from sklearn.metrics import f1_score, classification_report
import time
import sys
from torch.utils.data import Dataset, DataLoader
import warnings
from model import BiLSTM_CRF
from dataloader import get_data
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--DATA_PATH', type=str, default='data/')
parser.add_argument('--SAVE_CPT_PATH', type=str, default='checkpoint_tmp/')
parser.add_argument('--EMBEDDING_DIM', type=int, default=300)
parser.add_argument('--HIDDEN_DIM', type=int, default=200)
parser.add_argument('--BATCH_SIZE', type=int, default=16)
parser.add_argument('--DEV_BATCH_SIZE', type=int, default=256)
parser.add_argument('--EPOCHS', type=int, default=10)
parser.add_argument('--LR', type=float, default=0.01)
parser.add_argument('--Weight_Decay', type=float, default=1e-4)
parser.add_argument('--GPU_DEVICE', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_DEVICE
DATA_PATH = args.DATA_PATH
EMBEDDING_DIM = args.EMBEDDING_DIM
HIDDEN_DIM = args.HIDDEN_DIM
BATCH_SIZE = args.BATCH_SIZE
DEV_BATCH_SIZE = args.DEV_BATCH_SIZE
num_epochs = args.EPOCHS
LR = args.LR
Weight_Decay = args.Weight_Decay
SAVE_CPT_PATH = args.SAVE_CPT_PATH
# TRAIN_SIZE=50000

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
# 建立一个filehandler来把日志记录在文件里，级别为debug以上
fh = logging.FileHandler("log_{time}.log".format(time=str(time.time())))
fh.setLevel(logging.DEBUG)
# 建立一个streamhandler来把日志打在CMD窗口上，级别为error以上
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(lineno)s %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# 将相应的handler添加在logger对象中
logger.addHandler(ch)
logger.addHandler(fh)
# 开始打日志
logger.debug("debug message")
logger.info("info message")
logger.warn("warn message")
logger.error("error message")
logger.critical("critical message")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
START_TAG = "<START>"
STOP_TAG = "<STOP>"

torch.manual_seed(1)

loader, word_to_ix, tag_to_ix, word_set = get_data(DATA_PATH, BATCH_SIZE, DEV_BATCH_SIZE)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=Weight_Decay)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
# 开始训练
logger.info("=================================【Begin Training】=================================")
step = 0
for epoch in trange(num_epochs):
    logger.info("=================================【Begin Epoch_{}】=================================".format(str(epoch)))
    model.train()
    for batch_x, batch_y in tqdm(loader):
        logger.info(str(epoch) + '/' + str(num_epochs) + ':' + ' ' + str(step) + '/' + str(len(word_set) // BATCH_SIZE))
        step = step + 1
        model.zero_grad()
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        loss = model.neg_log_likelihood(torch.tensor(batch_x, dtype=torch.long),
                                        torch.tensor(batch_y, dtype=torch.long))
        # print(loss)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            logger.info("\n" + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) + \
                        ", epoch: " + str(epoch) + ", step: " + str(step) + ", loss: " + str(float(loss)))
        if step % 100 == 0:
            torch.save(model, SAVE_CPT_PATH+'checkpoint_{num}.pkl'.format(bsz=BATCH_SIZE, num=step))
            logger.info("Checkpoint has saved as checkpoint_{}.pkl in ./checkpoints".format(step))

    torch.save(model, SAVE_CPT_PATH+'checkpoint_{num}.pkl'.format(bsz=BATCH_SIZE, num=epoch))
    logger.info("Checkpoint has saved as checkpoint_{}.pkl in ./checkpoints".format(epoch))
    # 记录loss值
    loss = loss.cpu().detach().tolist()
    logger.info("epoch: " + str(epoch) + ", loss: " + str(float(loss)))
    '''
    logger.info("=================================【Evaluating】=================================")
    model.eval()
    all_pre = []
    all_tag = []
    for dev_batch_data, dev_batch_tag in tqdm(dev_dataloader):
        label = model.predict(dev_batch_data.view(len(dev_batch_data), -1).cuda())
        label = flatten(label)
        #xx,yy = model.forward(dev_batch_data.cuda())
        all_pre.extend(label)
        all_tag.extend(dev_batch_tag.detach().cpu().numpy().reshape(-1).tolist())
     # 记录loss值
    #loss = loss
    logger.info("epoch: " + str(epoch) + ", loss: " + str(float(loss)))
    org_no_O_PAD = [x for x, y in zip(all_tag, all_pre) if 1 <= x <= 8]
    pred_no_O_PAD = [y for x, y in zip(all_tag, all_pre) if 1 <= x <= 8]
    score = f1_score(org_no_O_PAD, pred_no_O_PAD, average="micro")
    logger.info(f"f1_score_1:{score:.3f}")
    org_no_O = [x for x, y in zip(all_tag, all_pre) if 1 <= x]
    pred_no_O = [y for x, y in zip(all_tag, all_pre) if 1 <= x]
    score = f1_score(org_no_O, pred_no_O, average="micro")
    logger.info(f"f1_score_2:{score:.3f}")
    org_no_PAD = [x for x, y in zip(all_tag, all_pre) if x <= 8]
    pred_no_PAD = [y for x, y in zip(all_tag, all_pre) if x <= 8]
    score = f1_score(org_no_O, pred_no_O, average="micro")
    logger.info(f"f1_score_3:{score:.3f}")
    score = f1_score(all_tag, all_pre, average="micro")
    logger.info(f"f1_score_4:{score:.3f}")
    '''

torch.save(model, SAVE_CPT_PATH+'bilstm_crf.pkl')
