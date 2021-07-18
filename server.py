import re
import urllib

import flask
from flask import render_template, request, jsonify
import json
import pandas as pd
import torch.nn as nn
import texthero as hero
from urllib.parse import urlsplit
import requests
import spacy_sentence_bert
import torch
from bs4 import BeautifulSoup
from requests import get
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import XLNetTokenizer, XLNetForSequenceClassification

from search import TAG_RE

print('Loading Similarity Model ... ')
nlp = spacy_sentence_bert.load_model('en_nli_roberta_base')
#
print('Loading Classification Model ... ')
tokenizer = XLNetTokenizer.from_pretrained('fakenews')
#
model = XLNetForSequenceClassification.from_pretrained("fakenews", num_labels=2,
                                                                   output_attentions=False,
                                                                   output_hidden_states=True)
# model.to(device)




def remove_tags(text):
    return TAG_RE.sub('', text)


def search(term, num_results=10, lang="en"):
    usr_agent = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0'}
        # 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
        #               'Chrome/61.0.3163.100 Safari/537.36'}

    def fetch_results(search_term, number_results, language_code):
        escaped_search_term = search_term.replace(' ', '+')

        google_url = 'https://www.google.com/search?q={}&num={}&hl={}'.format(escaped_search_term, number_results+1,
                                                                              language_code)
        response = get(google_url, headers=usr_agent)
        response.raise_for_status()

        return response.text

    def parse_results(raw_html):
        soup = BeautifulSoup(raw_html, 'html.parser')
        result_block = soup.find_all('div', attrs={'class': 'g'})
        for result in result_block:
            link = result.find('a', href=True)
            title = result.find('h3')
            text = result.findAll('span')[-1]
            # print(text.text)

            if link and title:
                yield link['href'], text.text

    html = fetch_results(term, num_results, lang)
    return list(parse_results(html))


def get_sentences_from_link(link, text, top=3):
    request = requests.get(link, verify=False, timeout=20)
    # time.sleep(1)
    Soup = BeautifulSoup(request.text, 'lxml')
    if 'twitter.com' in request.url:
        return [], request.url

    if 'facebook.com' in request.url:
        return [], request.url

    if '%PDF-' in request.text:
        return [], request.url

    # creating a list of all common heading tags
    heading_tags = ['h{}'.format(h) for h in range(1, 10)] + ['p']
    results = []
    used = []

    for tags in Soup.find_all(heading_tags):
        if 'h' in tags.name:
            tokens = tags.text.strip().split()
            if len(tokens) > 8:
                if tags.text.strip() not in used:
                    used.append(tags.text.strip())
                    results.append([tags.name, tags.text.strip()])
        else:
            tokens = tags.text.strip().split()
            if len(tokens) > 8:
                if tags.text.strip() not in used:
                    used.append(tags.text.strip())
                    results.append([tags.name, tags.text.strip()])
    doc1 = nlp(text)
    sim = []
    for r in results:
        sim.append(doc1.similarity(nlp(r[1])))
    zipped = zip(sim, results)
    zipped = sorted(zipped, reverse=True)
    high_conf = [a for s, a in zipped if s >= 0.5]

    return high_conf[:top], request.url


def filter_data(text, res, top=5):
    doc1 = nlp(text)
    sim = []
    for a in res:
        sim.append(doc1.similarity(nlp(a[1])))
    # print(sim)
    zipped = zip(sim, res)
    zipped = sorted(zipped, reverse=True)
    high_conf = [a for s, a in zipped if s >= 0.9]
    low_conf = [a for s, a in zipped if 0.7 <= s < 0.9]
    return high_conf, low_conf


def get_search_results(text, res, top=3):
    text_url = urllib.parse.quote_plus(text)
    res = search(text_url)
    blacklisted_phras= ['.pdf', '.xlsx', '.csv', '/download', 'facebook.com', 'youtube.com', 'patrika.com',
                        'maharashtratimes.com', 'books.google', '.txt', '.vocab']
    filtered_res = []
    for r in res:
        found = False
        for b_url in blacklisted_phras:
            if b_url in r[0]:
                found = True
                break
        if not found:
            filtered_res.append(r)
    high_conf, low_conf = filter_data(text, filtered_res)
    # print('High Confidence')
    # for r in enumerate(high_conf):
    #     print(r)
    # print('Low Confidence')
    # for r in enumerate(low_conf):
    #     print(r)
    if high_conf:
        return high_conf
    else:
        return low_conf[:top]


def get_evidence_links(text):
    links = []
    try:
        if 't.co/' in text:
            urls = re.findall(r'https?:\/+\/+t+\.+co+\/+\S*', text)
            new_links = []
            for li in urls:
                li = li.replace('.%20', '').replace('%20', '').strip('.').strip()
                if li[-1] == '.':
                    li = li[:-1]
                conf, lin = get_sentences_from_link(li, text)
                new_links.append([lin, conf])
            links = new_links

        else:
            links = get_search_results(text, links)
            new_links = []
            for link in links:
                conf, lin = get_sentences_from_link(link[0], text)
                new_links.append([lin, conf])
            links = new_links
    except:
        pass
    return links


def prepare_summary(claim, links):
    summ = []
    for link in links:
        if type(link[1]) == list:
            for text in link[1]:
                if type(link[1]) == list:
                    summ.append(text[1])
                else:
                    summ.append(text)
        elif type(link[1]) == str:
            summ.append(link[1])

            # print(link)
    # claim = text
    urls = re.findall(r'https?:\/+\/+t+\.+co+\/+\S*', claim)
    # new_links = []
    for li in urls:
        claim = claim.replace(li, '')
    claim = claim.strip()

    if summ:
        summary = 'Claim= {} Text= {}'.format(claim, ' '.join(summ).replace('\n', '').replace('\t', ''))
        # summary = '{}'.format(' '.join(summ).replace('\n', '').replace('\t', ''))
    else:
        summary = ''
        # c+=1
    return summary


class Summarizer:

    def __init__(self):
        print('Loading Summarizer Model ... ')
        self.tokenizer = T5Tokenizer.from_pretrained('claim_summ')
        self.model = T5ForConditionalGeneration.from_pretrained('claim_summ')

        # if USE_GPU:
        # self.model.to('cuda')

    def generate_summary(self, sentence):
        input_ids = self.tokenizer.encode('summarize: {}'.format(sentence), truncation=True)
        tokens_tensor = torch.tensor([input_ids])

        # generate text until the output length (which includes the context length) reaches 50
        generated_ids = self.model.generate(input_ids=tokens_tensor,
                                            max_length=250,
                                            num_beams=5,
                                            no_repeat_ngram_size=10,
                                            repetition_penalty=2.0)

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


summarizer = Summarizer()


def get_summary(evidence):
    return summarizer.generate_summary(evidence)


def classify(text):
    encoded_review = tokenizer.encode_plus(
      text,
      max_length=512,
      add_special_tokens=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']

    output = model(input_ids, attention_mask)
    # print(output)
    _, prediction = torch.max(output[0], dim=1)
    # print(f'Review text: {text}')
    confidence = nn.Softmax()(output[0])[0][prediction.item()].item()
    return {0: True, 1: False}[prediction.item()], confidence

def get_orgi_url(url):
    try:
        r = requests.get(url)
        return r.url
    except:
        return ''


def preprocess(text):
    s = pd.Series(text)
    s = hero.remove_diacritics(s)
    s = hero.remove_whitespace(s)
    s = s.tolist()[0]
    return s


def prepare_csv(text, links, summary):
    final_text = ''
    if not links:
        if 't.co' in text:
            urls = re.findall(r'https?:\/+\/+t+\.+co+\/+\S*', text)
            used = []
            for li in urls:
                text = text.replace(li, '')
                orig = get_orgi_url(li)
                base_url = "{0.scheme}://{0.netloc}/".format(urlsplit(orig))
                if 'twitter.com' in orig:
                    base_url = '/'.join(orig.split('/')[:4])
                base_url = base_url.replace('https://', '').replace('http://', '').replace('www.', '')[:-1]
                if base_url not in used:
                    # data['text'] += base_url + ' '
                    used.append(base_url)
            text += ' SOURCES: '
            for u in used:
                text += u + ' '

            # claim = claim.strip()
            final_text = text + ' [SEP] NA'
        else:
            final_text = text + ' SOURCES:  [SEP] NA'

    else:
        used = []
        for link in links:
            link = link[0]
            base_url = "{0.scheme}://{0.netloc}/".format(urlsplit(link))
            if 'twitter.com' in link:
                base_url = '/'.join(link.split('/')[:4])
            base_url = base_url.replace('https://', '').replace('http://', '').replace('www.', '')[:-1]
            if base_url not in used:
                # data['text'] += base_url + ' '
                used.append(base_url)
        text += ' SOURCES: '
        for u in used:
            text += u + ' '
        final_text = text + ' [SEP] ' + summary

    final_text = final_text.replace('Text=', ' ').replace('Claim=', ' ')
    if 't.co' in final_text:
        urls = re.findall(r'https?:\/+\/+t+\.+co+\/+\S*', final_text)
        for u in urls:
            if 't.co' in u:
                final_text = final_text.replace(u, ' ')

    final_text = preprocess(final_text)
    return final_text


app = flask.Flask(__name__)
app.config['DEBUG'] = True


@app.route('/', methods=['GET'])
def homepage():
    return render_template('homepage.html')


@app.route('/predict', methods=['POST'])
def background_process_test():
    json_data = request.json
    text = json_data['text']
    links = get_evidence_links(text)
    evidence = prepare_summary(text, links)
    summary = get_summary(evidence)
    final = prepare_csv(text, links, summary)
    prediction, confidence = classify(final)
    sources = final.split('SOURCES:')[1].strip().split('[SEP]')[0].strip()
    evidence = final.split('[SEP]')[1].strip()
    # tokens = final.split('[SEP]')
    # print(tokens)
    response_data = {'status': 'OK', 'prediction': prediction, 'sources': sources, 'evidence': evidence, 'confidence': confidence}
    # print(response_data)
    return json.dumps(response_data)


if __name__ == '__main__':
    app.run(port=7000, host='0.0.0.0')