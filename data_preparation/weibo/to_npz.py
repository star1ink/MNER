import json
import numpy as np
from collections import Counter
from utils.utils import deal_seq



# 数据处理
def Data_preprocess(input_filename, output_filename, mode="F"):
    count = 0
    # word, label, racial, seg_list, seg_loc_list
    word_list = []
    label_list = []
    racial_list = []
    seg_loc_list = []
    temps = []
    pure_words=[]
    jin_words = []
    data_json = open(input_filename, 'r', encoding="utf8")
    data = json.load(data_json)
    for sample in data:
        text, labels = sample['text'], sample['labels']
        words, labels, racial, seg_loc, t, pure_word = deal_seq(text,labels)
        if len(labels) == 0 :
            continue
        word_list.append(words)
        label_list.append(labels)
        racial_list.append(racial)
        seg_loc_list.append(seg_loc)
        temps.append(t)
        pure_words.append(pure_word)
    # 保存成二进制文件'
    #print(word_list[0], label_list[0], racial_list[0], seg_loc_list[0], temps[0], pure_words[0])
    #print(len(word_list),len(label_list),len(racial_list),len(seg_loc_list),len(temps),len(pure_words))
    np.savez_compressed(output_filename, words=word_list, lables=label_list, racials=racial_list,
                        seg_loc = seg_loc_list,temps = temps,pure_words=pure_words)
    print(1)
    print(len(word_list))
    # 统计处理数量
    label_name = ['PER.NAM', 'PER.NOM', 'LOC.NAM', 'GPE.NAM', 'ORG.NOM', 'LOC.NOM', 'ORG.NAM']
    sum_label = label_list[0]
    for i in range(len(label_list) - 1):
        sum_label.extend(label_list[i + 1])
    result = Counter(sum_label)
    result['counter'] = len(label_list)
    for name in label_name:
        result.pop('I-' + name)
    print(result)
    return None


datanames = ["weiboNER."]
sets = ["dev.", "test.", "train."]
for dataname in datanames:
    for name in sets:
        Data_preprocess(dataname + name + "json", dataname + name + "npz")
