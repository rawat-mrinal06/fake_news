# from research.database.feverous_db import FeverousDB
import csv
import json
import nltk
import spacy_sentence_bert
from tqdm import tqdm
from research.fever_db import FeverDB


def read_json():
    data = []
    with open('fever/fever/train.jsonl') as f:
        for l in f:
            d = json.loads(l.strip())
            data.append(d)
    return data


def get_text_from_db(id_no):
    db = FeverDB('fever/fever/wiki_fever.db')
    page_json = db.get_doc_text(id_no)
    return page_json


def get_from_db(id_no, sen_no):
    db = FeverDB('fever/fever/wiki_fever.db')
    page_json = db.get_doc_text(id_no)
    sentences = nltk.sent_tokenize(page_json)
    return sentences[sen_no]


nlp = spacy_sentence_bert.load_model('en_nli_roberta_base')


def filter_data(text, res, top=7):
    doc1 = nlp(text)
    sim = []
    for a in res:
        sim.append(doc1.similarity(nlp(a[1])))
    # print(sim)
    zipped = zip(sim, res)
    zipped = sorted(zipped, reverse=True)
    # high_conf = [a for s, a in zipped if s >= 0.9]
    conf = [a for s, a in zipped if s >= 0.25]
    return conf[:10]


def prepare_summary(data):
    f = open('train_fever1.csv', 'w')
    wr = csv.writer(f)
    wr.writerow(['text', 'summ'])
    for d in tqdm(data):
        try:
            summ = []
            used_articles = []
            text = []
            if d['label'] == 'SUPPORTS':
                for evidence in d['evidence']:
                    for e in evidence:
                        if e[2] not in used_articles:
                            used_articles.append(e[2])
                            text.append(get_text_from_db(e[2]))

                        summ.append(get_from_db(e[2], e[3]))
                if 8 < len(nltk.sent_tokenize(''.join(text))) <= 15  and len(summ) > 2:
                    # text = filter_data(d['claim'], nltk.sent_tokenize(''.join(text)))
                    # text = filter_data(d['claim'], nltk.sent_tokenize(''.join(text)))
                # else:
                #     text = nltk.sent_tokenize(''.join(text))
                    wr.writerow(['Claim= ' + d['claim'] + ' Text= ' + ''.join(text), ''.join(summ)])
                # break
        except:
            pass
            # print('Claim: ', d['claim'])
            # print('Text: ', ''.join(text))
            # print('Summ: ', ''.join(summ))
            # break


if __name__ == '__main__':
    prepare_summary(read_json())
