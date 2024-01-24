import json

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import conf
from transformers import pipeline
#tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
#model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
#nlp = pipeline("ner", model=model, tokenizer=tokenizer)
#example = "Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute."
#ner_results = nlp(example)
#print(ner_results)


class NERDataset(Dataset):
    def __init__(self, words, labels,radicals,temps, word_pad_idx=0, label_pad_idx=-1):
        # 词元模块
        self.tokenizer = AutoTokenizer.from_pretrained(conf.roberta_model)
        # 将标签映射为id，再反向映射
        self.label2id = conf.label2id
        self.id2label = {_id: _label for _label, _id in list(conf.label2id.items())}
        # 获得数据（词元->id）
        self.dataset = self.preprocess(words, labels,radicals,temps)
        # 词元填充id
        self.word_pad_idx = word_pad_idx
        # 标签填充id
        self.label_pad_idx = label_pad_idx
        self.device = conf.device

    def preprocess(self, origin_sentences, origin_labels,ori_radical,ori_temp):
        # 初始化
        data = []
        #sentences = []
        labels = []
        radicals = ori_radical
        temps = []
        # 使用进度条遍历seq
        for ori_sentences,temp,tag,radical in tqdm(zip(origin_sentences,ori_temp,origin_labels,ori_radical)):
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            # 初始化字列表
            # words = []
            # word_lens = []
            #for token in ori_sentences:
                # bert对字进行编码转化为id表示，有的可能拆出多个词元
                # words.append(self.tokenizer.tokenize(token))
                # 记录每个字的词元长度，一般为1
                # word_lens.append(len(token))
            # 变成单个字的列表，开头加上[CLS]
            #words = ['[CLS]'] + [item for token in words for item in token]
            # 记录每个字首个词元的索引，从1开始，跳过CLS
            # 因为一般每个字词元长度为1，所以大多为自增情况，长度为序列长度
            #token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            # 将words转为ids，并与词元开头的索引放在一起
            # sentences.append()
            label_id = [self.label2id.get(t) for t in tag]
            data.append(((self.tokenizer.convert_tokens_to_ids(ori_sentences), temp), label_id,radical))
        # 遍历处理tag
        # for tag in origin_labels:
            # 将label转到id
        #    label_id = [self.label2id.get(t) for t in tag]
        #    labels.append(label_id)
        # 一致性检验，句子长度等于label长度-1
        # for sentence, label in zip(sentences, labels):
        #    if len(sentence[0]) - len(label) == 1:
        # data.append((sentence, label,radicals))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        radical = self.dataset[idx][2]
        #temp = self.dataset[idx][3]
        return [word, label,radical]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        # 解出seq与label
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        radicals = [x[2] for x in batch]
        # batch大小
        batch_len = len(sentences)
        seq_len = [len(s[0]) for s in sentences]

        # 得到这一batch的最大seq长度
        max_len = max([len(s[0]) for s in sentences])
        # 初始化最大label长度为0
        max_label_len = 0  # 改动前max_label_len = 0
        # padding data 初始化，维度为（batch size， max len）
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        # 批数据label起始位置初始化
        batch_label_starts = []
        # 填充对齐
        for j in range(batch_len):
            # 使用当前样本的有效seq id替代pad
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            # seq的-1维度为起始位置信息
            # 找到有标签的数据的index（[CLS]不算），也就是字对应的的首个词元索引，对应的，也就是该字的label索引，非对应值为0，也就是label的pad
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            # 计算有效的label数量最大值
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # pad label，维度为（batch size，label len）
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        batch_radicals = self.label_pad_idx * np.ones((batch_len, max_label_len))
        # 有效label数据替代
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j]
            batch_radicals[j][:cur_tags_len] = radicals[j]
        # 有效label数据替代
        # data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        batch_radicals = torch.tensor(batch_radicals, dtype=torch.long)
        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch_seq_len = torch.tensor(seq_len,dtype = torch.int).to(self.device)
        batch_radicals = torch.tensor(batch_radicals, dtype=torch.int).to(self.device)
        return [batch_data, batch_label_starts, batch_labels,batch_radicals,batch_seq_len]

def read_json(filename):
    word,label = [],[]
    with open(filename, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        for item in json_data:
            word.append(''.join(item['text']))
            label.append(item['labels'])
    return word,label

def read_npz(filename):
    data = np.load(filename,allow_pickle=True)
    return [data[name] for name in data.files]

if __name__ =='__main__':
    word_dev, label_dev, radiacl_dev,seg_loc_dev,temps_dev,pure_words_dev = read_npz('../../data_preparation/weibo/weiboNER.train.npz')
    nerdata = NERDataset(word_dev,label_dev,radiacl_dev,temps_dev)
    print(nerdata[10])
    train_loader = DataLoader(nerdata, batch_size=conf.batch_size,
                              shuffle=True, collate_fn=nerdata.collate_fn)
    for item in train_loader:
        print(item)
        break


