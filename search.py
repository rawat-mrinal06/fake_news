import csv
import json
import urllib

import requests
import spacy
import tqdm
from requests import get
from bs4 import BeautifulSoup
import re
import spacy_sentence_bert


TAG_RE = re.compile(r'<[^>]+>')


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


# nlp = spacy.load('en_core_web_lg')
nlp = spacy_sentence_bert.load_model('en_nli_roberta_base')


def filter_sites():
    pass


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
    # text_url = urllib.parse.quote_plus(text)
    # res = search(text_url)
    blacklisted_phras= ['.pdf', '.xlsx', '.csv', '/download', 'facebook.com', 'youtube.com', 'patrika.com', 'maharashtratimes.com', 'books.google', '.txt', '.vocab']
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
    print('High Confidence')
    for r in enumerate(high_conf):
        print(r)
    print('Low Confidence')
    for r in enumerate(low_conf):
        print(r)
    if high_conf:
        return high_conf
    else:
        return low_conf[:top]


def prepare():
    f = open('test.csv')
    f_w = open('test_links.json', 'a')
    reader = csv.reader(f)
    data = []
    for r in reader:
        data.append(r)
    data = data[1:]
    for i, d in enumerate(tqdm.tqdm(data)):
        if i >= 0:
            try:
                if 't.co/' not in d[1]:
                    text_url = urllib.parse.quote_plus(d[1])
                    res = search(text_url)
                    js = {'id': d[0], 'text': d[1],  'links': res}
                    f_w.write(json.dumps(js) + '\n')
                else:
                    js = {'id': d[0], 'text': d[1],  'links': []}
                    f_w.write(json.dumps(js) + '\n')
            except:
                js = {'id': d[0], 'text': d[1],  'links': []}
                f_w.write(json.dumps(js) + '\n')

    f.close()
    f_w.close()


def check():
    cnt = 1
    with open('train_links1.json') as f:
        for l in f:
            data = json.loads(l.strip())
            if not data['links'] and 't.co/' not in data['text']:
                print(data)
            cnt += 1
            # if cnt == 2010:
            #     break


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


def get_sentences():
    datas = []
    with open('test_links.json') as f:
        for l in f:
            data = json.loads(l.strip())
            datas.append(data)
    with open('test_links1.json', 'a') as f:
        for i, data in enumerate(tqdm.tqdm(datas)):
            if i >= 229:
                try:
                    if 't.co/' in data['text']:
                        urls = re.findall(r'https?:\/+\/+t+\.+co+\/+\S*', data['text'])
                        new_links = []
                        for li in urls:
                            li = li.replace('.%20', '').replace('%20', '').strip('.').strip()
                            if li[-1] == '.':
                                li = li[:-1]
                            conf, lin = get_sentences_from_link(li, data['text'])
                            new_links.append([lin, conf])
                        data['links'] = new_links
                        f.write(json.dumps(data) + '\n')
                    else:
                        # print(data['text'])
                        links = get_search_results(data['text'], data['links'])
                        new_links = []
                        for link in links:
                            conf, lin = get_sentences_from_link(link[0], data['text'])
                            new_links.append([lin, conf])
                        data['links'] = new_links
                        f.write(json.dumps(data) + '\n')
                except:
                    f.write(json.dumps(data) + '\n')
                    # break


if __name__ == '__main__':
    # pass
    # get_sentences()
    # text = 'The CDC currently reports 99031 deaths. In general the discrepancies in death counts between different sources are small and explicable. The death toll stands at roughly 100000 people today.'
    # link = 'https://www.cdc.gov/nchs/nvss/vsrr/covid19/tech_notes.htm'
    # get_search_results(text,r)
    # conf, links = get_
    get_sentences()

    # get_search_results(text)
    # prepare()
    # re_prepare()
    # check()