import logging
from transformers.optimization import get_cosine_schedule_with_warmup,AdamW
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
import conf
import torch.nn as nn
from sklearn.metrics import accuracy_score,recall_score,f1_score
from transformers import BertTokenizer
import torch
import numpy as np
#定义训练函数
from models.RoBERTa_finetuned2.BERTNER import BertNER
from models.RoBERTa_finetuned2.DataSet import NERDataset



log_path = "D:\\study\\yanyi\\NER\\models/RoBERTa_finetuned2/Log/"
logger.add(log_path + 'Train.log', format="{time} {level} {message}", level="INFO")
def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    # 设定训练模式
    model.train()
    train_losses = 0
    loss_func = nn.CrossEntropyLoss(ignore_index = -1)
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels,batch_radicals, seq_len = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask
        # 计算损失值
        pred = model((batch_data, batch_token_starts),batch_radicals,
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
        # loss_val = loss_func(pred[-1],batch_labels)

        train_losses += pred[0]
        # 梯度更新
        model.zero_grad()
        pred[0].backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=conf.clip_grad)
        # 计算梯度
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    logger.info("Epoch: {}, train loss: {}",epoch, train_loss)

#根据预测值和真实值计算评价指标
def compute_acc_recall(batch_output,batch_tags):
    acc = 0
    recall = 0
    f1 = 0
    for index in range(len(batch_output)):
        acc += accuracy_score(batch_output[index],batch_tags[index])
        recall += recall_score(batch_output[index],batch_tags[index],average='macro')
        f1 += f1_score(batch_output[index],batch_tags[index],average='macro')
    return (acc/len(batch_output),recall/len(batch_output),f1/len(batch_output))
#定义验证函数
# 评价集合判断？？？
def evaluate(dev_loader, model, mode='dev'):
    # 设置为模型为验证模式
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(conf.roberta_model, do_lower_case=True, skip_special_tokens=True)
    id2label = conf.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    with torch.no_grad():
        for idx, batch_samples in tqdm(enumerate(dev_loader)):
            batch_data, batch_token_starts, batch_labels,batch_radicals, seq_len = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_labels.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            #print(label_masks)
            #print(batch_labels)
            pred = model((batch_data, batch_token_starts),batch_radicals,
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
            # loss_val = loss_func(pred,batch_labels)
            # dev_losses += loss_val
            dev_losses += pred[0]
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),batch_radicals,
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            batch_labels = batch_labels.to('cpu').numpy()
            pred_tags.extend([[idx for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[idx for idx in indices if idx > -1] for indices in batch_labels])
            #pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            #true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
    assert len(pred_tags) == len(true_tags)
    # logging loss, f1 and report
    metrics = {}
    acc , recall, F1= compute_acc_recall(true_tags,pred_tags)
    metrics['acc'] = acc
    metrics['recall'] = recall
    metrics['f1'] = F1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics

def test(conf):
    word_test, label_test, radiacl_test, seg_loc_test, temps_test, pure_words_test = read_npz("../../data_preparation/weibo/weiboNER.test.npz")
    test_dataset = NERDataset(word_test, label_test, conf)
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    # Prepare model
    if conf.model_dir is not None:
        model = torch.load(conf.model_dir)
        #model = BertNER.from_pretrained(conf.model_dir)
        model.to(conf.device)
    val_metrics = evaluate(test_loader, model, mode='test')
    logging.info("test loss: {}, f1 score: {},acc: {},recall: {}".format(val_metrics['loss'], val_metrics['F1'],val_metrics['acc'],val_metrics['recall']))

def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, conf.epoch_num + 1):
        # train_epoch(train_loader, model, optimizer, scheduler, epoch)
        #开始验证
        val_metrics = evaluate(dev_loader, model, mode='dev')
        val_f1 = val_metrics['f1']
        logger.info("Epoch: {}, dev loss: {}, f1 score: {}",epoch, val_metrics['loss'], val_f1)
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            #模型保存需要更改
            torch.save(model,model_dir)
            logger.info("--------Save best model!--------")
            if improve_f1 < conf.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= conf.patience_num and epoch > conf.min_epoch_num) or epoch == conf.epoch_num:
            logger.info("Best val f1: {}",best_val_f1)
            break
    logger.info("Training Finished!")

def run(config):
    """train the model"""
    # 处理数据，
    # 分离训练集、验证集
    word_train, label_train,radiacl_train,seg_loc_train,temps_train,pure_words_train = read_npz("../../data_preparation/weibo/weiboNER.train.npz")
    word_dev, label_dev, radiacl_dev,seg_loc_dev,temps_dev,pure_words_dev = read_npz("../../data_preparation/weibo/weiboNER.dev.npz")
    # 创建dataset
    train_dataset = NERDataset(word_train, label_train,radiacl_train, temps_train)
    dev_dataset = NERDataset(word_dev, label_dev,radiacl_dev, temps_dev)
    # get dataset size
    train_size = len(train_dataset)
    # 创建dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)

    # 实例化模型
    device = config.device
    model = BertNER.from_pretrained(config.roberta_model, num_labels=conf.num_labels,ignore_mismatched_sizes=True)
    model.to(device)
    # Prepare optimizer
    bert_optimizer = list(model.bert.named_parameters())
    lstm_optimizer = list(model.bilstm.named_parameters())
    radicalfea_optimizer = list(model.radical_features.named_parameters())
    classifier_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in radicalfea_optimizer if not any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in radicalfea_optimizer if any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': 0.0},
        {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': 0.0},
        {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': 0.0},
        {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
    ]
    if not config.full_fine_tuning:
        optimizer_grouped_parameters = optimizer_grouped_parameters[2:]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)


#定义推断函数
def infer_function(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(conf.roberta_model, do_lower_case=True, skip_special_tokens=True)
    id2label = conf.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0
    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            #loss = model((batch_data, batch_token_starts),
                         #token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            #dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            #batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
    return pred_tags

def new_infer(text):

    words = list(text)
    label = ['O'] * len(words)
    word_list = []
    label_list = []
    word_list.append(words)
    label_list.append(label)
    output_filename = 'D:\\study\\yanyi\\NER\\infer.npz'
    np.savez_compressed(output_filename,words = word_list, lables = label_list)
    #重新加载
    data = np.load(output_filename, allow_pickle=True)
    word_test = data["words"]
    label_test = data["lables"]
    test_dataset = NERDataset(word_test, label_test, conf)
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    # Prepare model
    if conf.model_dir is not None:
        #model = torch.load(NER_config.model_dir)
        model = BertNER.from_pretrained(conf.model_dir)
        model.to(conf.device)
        logger.info("--------Load model from {}--------".format(conf.model_dir))
    else:
        logger.info("--------No model to test !--------")
        return
    pre_tegs = infer_function(test_loader, model, mode='test')
    return pre_tegs





def read_npz(filename):
    data = np.load(filename,allow_pickle=True)
    return [data[name] for name in data.files]

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    run(conf)
    model = BertNER.from_pretrained(conf.roberta_model, num_labels=15, ignore_mismatched_sizes=True)
    model.to(conf.device)
    print(model)
    for name, parameters in model.named_parameters():  # 打印出每一层的参数的大小
        print(name, ':', parameters.size())
    print(get_parameter_number(model))
    #
    test(conf)
    # text = '2022年11月，马拉西亚随荷兰国家队征战2022年卡塔尔世界杯'
    # pre_tegs = new_infer(text)
    #
    # # 取出位置
    # start_index_list = []
    # end_index_list = []
    # for index in range(len(pre_tegs[0])):
    #     if index != 0 and pre_tegs[0][index] != 'O' and pre_tegs[0][index - 1] == 'O':
    #         start_index = index
    #         start_index_list.append(start_index)
    #     if index != len(pre_tegs[0]) - 1 and pre_tegs[0][index] != 'O' and pre_tegs[0][index + 1] == 'O':
    #         end_index = index
    #         end_index_list.append(end_index)
    #     if index == 0 and pre_tegs[0][index] != 'O':
    #         start_index = index
    #         start_index_list.append(start_index)
    #     if index == len(pre_tegs[0]) - 1 and pre_tegs[0][index] != 'O':
    #         end_index = index
    #         end_index_list.append(end_index)
    # # 展示
    # for index in range(len(start_index_list)):
    #     print(text[start_index_list[index]:end_index_list[index] + 1])