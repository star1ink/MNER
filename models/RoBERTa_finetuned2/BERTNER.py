import torch
from transformers.models.bert.modeling_bert import *
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
import conf

class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        # 定义分类类别，也可以写在加载预训练模型的config文件中
        self.num_labels = conf.num_labels
        self.bert = BertModel(config)
        for p in self.parameters():
            p.requires_grad = False
        self.radical_features = nn.Embedding(conf.racials_num,conf.racials_embeds)
        self.dropout = nn.Dropout(conf.lstm_hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=conf.lstm_embedding_size,  # 1024
            hidden_size=conf.hidden_size // 2,  # 1024
            batch_first=True,
            num_layers=2,
            dropout=conf.lstm_dropout_prob,  # 0.5
            bidirectional=True
        )
        self.classifier = nn.Linear(conf.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_data,radicals, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        radicals_features = self.radical_features(radicals+1)
        # 将结果送入bilstm，再次提取特性
        #print(padded_sequence_output.shape,radicals_features.shape)
        lstm_output, _ = self.bilstm(torch.concat((padded_sequence_output,radicals_features),-1))
        # 将lstm的结果送入线性层，进行分类
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            # 将每个标签的概率送入到crf中进行解码，并获得loss
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs
        # contain: (loss), scores
        return outputs