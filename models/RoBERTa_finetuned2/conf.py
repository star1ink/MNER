import os
import torch

roberta_model = 'D:\\study\\yanyi\\NER\\pre_model\\RoBERTa_wwm_ext\\'
model_dir = 'D:\\study\\yanyi\\NER\\models/RoBERTa_wwm_ext/RoBERTa_wwm_ext_nnew.pkl'
# 是否加载训练好的NER模型
load_before = False
# 指定device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# 是否对整个BERT进行fine tuning
full_fine_tuning = False

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 8
epoch_num = 10
min_epoch_num = 3
patience = 0.0002
patience_num = 3

# label2id = {'O': 0,'B-PER': 1,'I-PER': 2,
#        'B-LOC': 3,'I-LOC': 4,'B-ORG': 5,'I-ORG': 6}
label2id = {'O': 0,'B-GPE.NAM': 1,'I-GPE.NAM': 2,
       'B-PER.NAM': 3,'I-PER.NAM': 4,'B-PER.NOM': 5,'I-PER.NOM': 6,
       'B-ORG.NOM': 7,'I-ORG.NOM': 8,'B-ORG.NAM': 9,'I-ORG.NAM': 10,
       'B-LOC.NOM': 11,'I-LOC.NOM': 12,'B-LOC.NAM': 13,'I-LOC.NAM': 14}
id2label = {_id: _label for _label, _id in list(label2id.items())}

# BertNER的超参数,也可以设置在预训练模型的config中

device = 'cpu'
num_labels = len(label2id)
hidden_dropout_prob = 0.3
lstm_hidden_dropout_prob = 0.3
racials_embeds = 100
lstm_embedding_size = 768 + racials_embeds
hidden_size = 1024
lstm_dropout_prob = 0.4
racials_num = 222+1




racial_li = ['卜', '巛', '行', '屮', '艸', '又', '廴', '言', '匸', '衣', '厶', '口', '入', '匕', '人', '龠', '龍', '广', '丨', '十', '一', '儿', '乙', '二', '羊', '亠', '讠', '火', '禸', '竹', '彳', '辵', '爿', '齒', '齊', '鼻', '鼠', '鼓', '黹', '黍', '黽', '黑', '鼎', '魚', '鳥', '麥', '麻', '鹿', '鹵', '黃', '角', '馬', '鬲', '高', '鬥', '鬯', '髟', '音', '頁', '香', '韋', '首', '面', '韭', '鬼', '骨', '革', '風', '隹', '食', '門', '隶', '金', '阜', '非', '長', '雨', '靑', '足', '走', '豸', '酉', '邑', '辛', '豕', '身', '里', '見', '谷', '豆', '赤', '示', '辰', '車', '釆', '貝', '糸', '羽', '自', '舟', '至', '聿', '血', '攴', '襾', '舌', '色', '肉', '米', '臼', '虍', '缶', '艮', '耳', '而', '虫', '厂', '臣', '耒', '玉', '用', '疋', '穴', '玄', '田', '网', '矢', '石', '生', '皮', '白', '疒', '毋', '矛', '立', '龜', '甘', '歹', '癶', '爪', '止', '目', '禾', '瓜', '皿', '支', '曰', '爻', '牙', '心', '文', '尢', '瓦', '水', '殳', '手', '冂', '日', '月', '氏', '犬', '欠', '气', '片', '牛', '老', '斤', '毛', '木', '宀', '无', '戶', '戈', '父', '方', '斗', '卩', '贝', '比', '车', '子', '夊', '弋', '幺', '舛', '囗', '己', '士', '土', '夕', '小', '夂', '冫', '山', '女', '彐', '廾', '弓', '工', '几', '巾', '彡', '尸', '丿', '干', '飛', '大', '寸', '凵', '力', '八', '匚', '阝', '冖', '刀', '勹', '亅', '丶', 'A', 'D', 'S','E']

data_name = "weibo"
data_set= {
       "weibo":{"train":"D:\\study\\yanyi\\NER\\data_preparation/weibo/weiboNER.train.npz",
                "dev":"D:\\study\\yanyi\\NER\\data_preparation/weibo/weiboNER.dev.npz",
                "test":"D:\\study\\yanyi\\NER\\data_preparation/weibo/weiboNER.test.npz"}
}
# mode: "O":原始句子，不加入SPA，“F”：加入SPA分词句子，"#"以#作为词内标记
mode = "O"