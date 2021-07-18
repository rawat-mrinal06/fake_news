import csv
import json
import re
from urllib.parse import urlsplit

import requests


def get_orgi_url(url):
    try:
        r = requests.get(url)
        return r.url
    except:
        return ''


def prepare_csv():
    f1 = open('test_summ.csv', 'a')
    wr = csv.writer(f1)
    wr.writerow(['tweet', 'label'])
    cnt = 0
    with open('data_test_summ.json') as f:
        for l in f:
            if cnt >= 1876:
                data = json.loads(l.strip())
                if not data['links']:
                    if 't.co' in data['text']:
                        urls = re.findall(r'https?:\/+\/+t+\.+co+\/+\S*', data['text'])
                        used = []
                        for li in urls:
                            data['text'] = data['text'].replace(li, '')
                            orig = get_orgi_url(li)
                            base_url = "{0.scheme}://{0.netloc}/".format(urlsplit(orig))
                            if 'twitter.com' in orig:
                                base_url = '/'.join(orig.split('/')[:4])
                            base_url = base_url.replace('https://', '').replace('http://', '').replace('www.', '')[:-1]
                            if base_url not in used:
                                # data['text'] += base_url + ' '
                                used.append(base_url)
                        data['text'] += ' SOURCES: '
                        for u in used:
                            data['text'] += u + ' '

                        # claim = claim.strip()
                        wr.writerow([data['text'] + ' <SEP> ', data['label']])
                    else:
                        wr.writerow([data['text'] + ' SOURCES: ' + ' <SEP> ', data['label']])
                else:
                    used = []
                    for link in data['links']:
                        link = link[0]
                        base_url = "{0.scheme}://{0.netloc}/".format(urlsplit(link))
                        if 'twitter.com' in link:
                            base_url = '/'.join(link.split('/')[:4])
                        base_url = base_url.replace('https://', '').replace('http://', '').replace('www.', '')[:-1]
                        if base_url not in used:
                            # data['text'] += base_url + ' '
                            used.append(base_url)
                    data['text'] += ' SOURCES: '
                    for u in used:
                        data['text'] += u + ' '

                    wr.writerow([data['text'] + ' <SEP> ' + data['summ'], data['label']])
            cnt += 1
    f1.close()


prepare_csv()