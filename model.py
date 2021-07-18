from copy import deepcopy

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchtext import data

from torchtext import datasets
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoModelForSequenceClassification, AutoTokenizer,RobertaForSequenceClassification,RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, BartForConditionalGeneration, BartTokenizer, AutoTokenizer, AutoModel, XLMRobertaForSequenceClassification, XLMRobertaTokenizer

batch_size = 2
cudnn.benchmark = True  # fire on all cylinders
epochs = 1

device = torch.device("cuda")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tokenizer = BertTokenizer.from_pretrained('deepset/covid_bert_base')
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
print('Loaded tokenizer')
max_input_length = 512
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

from sklearn.metrics import precision_score, recall_score, f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print(list(labels_flat))
    print(list(pred_flat))
    return np.sum(pred_flat == labels_flat) / len(labels_flat), precision_score(labels_flat, pred_flat),recall_score(labels_flat, pred_flat),f1_score(labels_flat, pred_flat)


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens


class BERTClassifier:

    def __init__(self):

        self.TEXT = data.Field(
            use_vocab=False,
            tokenize=tokenize_and_cut,
            preprocessing=tokenizer.convert_tokens_to_ids,
            init_token=init_token_idx,
            eos_token=eos_token_idx,
            pad_token=pad_token_idx,
            unk_token=unk_token_idx)

        self.LABEL = data.Field(sequential=False)

        self.train_iter, self.val_iter, self.test_iter, self.train, self.val, self.test = self.setup()

        self.model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2,
                                                                   output_attentions=False,
                                                                   output_hidden_states=True).cuda()

        # self.model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
        # self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).cuda()
        print('Loaded model')

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(self.train) * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=total_steps)

    def setup(self):
        train = data.TabularDataset(path='train_summ.csv',
                                    format='csv',
                                    fields=[('tweet', self.TEXT), ('label', self.LABEL)], skip_header=True)

        val = data.TabularDataset(path='val_summ.csv',
                                  format='csv',
                                  fields=[('tweet', self.TEXT), ('label', self.LABEL)], skip_header=True)

        test = data.TabularDataset(path='test_summ.csv',
                                   format='csv',
                                   fields=[('tweet', self.TEXT), ('label', self.LABEL)], skip_header=True)

        self.TEXT.build_vocab(train, max_size=10000)
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, val, test), sort_key=lambda x: len(x.tweet), batch_size=batch_size, repeat=False)
        return train_iter, val_iter, test_iter, train, val, test

    def train_model(self):
        self.model.train()
        data_loss_ema = 0
        oe_loss_ema = 0

        for batch_idx, batches in enumerate(iter(self.train_iter)):
            batch = batches

            inputs = batch.tweet.t().to(device)
            labels = (batch.label - 1).to(device)
            # print(labels)
            self.model.zero_grad()

            outputs = self.model(inputs)

            logits = outputs[0]

            data_loss = F.cross_entropy(logits, labels.cuda())

            loss = data_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            data_loss_ema = data_loss_ema * 0.9 + data_loss.data.cpu().numpy() * 0.1
            # if self.oe:
            #     oe_loss_ema = oe_loss_ema * 0.9 + oe_loss.data.cpu().numpy() * 0.1

            if batch_idx % 200 == 0 or batch_idx < 10:
                print('iter: {} \t| data_loss_ema: {} \t| oe_loss_ema: {}'.format(
                    batch_idx, data_loss_ema, oe_loss_ema))

            self.scheduler.step()

    def evaluate(self, data_iter):
        self.model.eval()
        eval_loss, eval_accuracy, eval_pr, eval_rec, eval_f1 = 0, 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch_idx, batch in enumerate(iter(data_iter)):
            inputs = batch.tweet.t().to(device)
            labels = (batch.label - 1).to(device)
            # print(labels)

            with torch.no_grad():
                outputs = self.model(inputs)

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            tmp_eval_accuracy, pr, rec, f1 = flat_accuracy(logits, label_ids)

            # loss = F.cross_entropy(logits, labels.cuda(), size_average=False)
            # running_loss += loss.data.cpu().numpy()

            eval_accuracy += tmp_eval_accuracy
            eval_pr += pr
            eval_rec += rec
            eval_f1 += f1
            # Track the number of batches
            nb_eval_steps += 1

        return eval_accuracy / nb_eval_steps, eval_pr / nb_eval_steps, eval_rec / nb_eval_steps, eval_f1 / nb_eval_steps

    def start_training(self, keyword):
        acc, pr, rec, f1 = self.evaluate(self.test_iter)
        print('test acc: {}, {}, {}, {}'.format(acc, pr, rec, f1))
        print('test acc: {}'.format(acc))
        best_f1 = 0
        for epoch in range(epochs):
            print('Epoch', epoch)
            self.train_model()
            acc, pr, rec, f1 = self.evaluate(self.val_iter)
            print('test acc: {}, {}, {}, {}'.format(acc, pr, rec, f1))
            writer.add_scalar("Accuracy", self.accuracy(), epoch)
            writer.add_scalar("Precision", self.accuracy(), epoch)
            writer.add_scalar("Recall", self.accuracy(), epoch)
            writer.add_scalar("F1-ccore", self.accuracy(), epoch)

            if f1 > best_f1:
                best_f1 = f1
                self.model.save_pretrained('models')
                tokenizer.save_pretrained('models')

        self.model.from_pretrained('models')
        tokenizer.from_pretrained('models')
        acc, pr, rec, f1 = self.evaluate(self.test_iter)
        print('test acc: {}, {}, {}, {}'.format(acc, pr, rec, f1))

        # torch.save(self.model.state_dict(), 'allmodels/{}/{}/model.dict'.format(self.in_domain, keyword))
        print('Saved model.')


classifier = BERTClassifier()
classifier.start_training('ES')
