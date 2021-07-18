import json
from urllib.parse import urlsplit

from tqdm import tqdm

from data.get_all_texts import filter_data
from data.get_evidences import get_top_k_results_from_google, get_relevant_text_from_webpage


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_data():
    data = []
    cnt = 0
    with open('data.json') as f:
        for l in f:
            d = json.loads(l.strip())
            if not d.get('res'):
                cnt += 1
                # print(d)
            data.append(d)
    print(cnt)
    return data

data = read_data()

# f = open('data1.json', 'w')
# idx = 6124
# for d in data:
#     d['idx'] = idx
#     idx += 1
#     f.write(json.dumps(d) + '\n')
#
# f.close()

def refind():
    data = read_data()
    f = open('data_train.json', 'a')
    for i, d in tqdm(enumerate(data)):
        text = d['text']
        if i >= 5925:
            if not d['res']:
                try:
                    res = []
                    if 'https://' not in text:
                        total_links = get_top_k_results_from_google(text, k=9)
                        links_chunks = list(divide_chunks(total_links, 3))

                        found = False
                        new_links = []
                        for links in links_chunks:
                            for l in links:
                                text_results, _ = get_relevant_text_from_webpage(l)
                                filtered_results = filter_data(text, text_results)
                                if filtered_results:
                                    new_links.append(l)
                                    res.append(filtered_results)
                                    found = True
                            if found:
                                break

                        f.write(json.dumps({'idx': i + 1, 'url': new_links, 'text': text, 'res': res}) + '\n')
                    else:
                        new_r = []
                        for r in d['res']:
                            if r:
                                temp = []
                                for ir in r:
                                    if 'JavaScript is disabled in this browser' in ir[1]:
                                        temp = []
                                        # new_r.append(r)
                                        break
                                    else:
                                        temp.append(ir)
                                new_r.append(temp)
                            else:
                                new_r.append(r)
                        found = False
                        for r in new_r:
                            if r:
                                found = True
                                break
                        if not found:
                            total_links = get_top_k_results_from_google(text, k=9)
                            links_chunks = list(divide_chunks(total_links, 3))

                            found = False
                            new_links = []
                            for links in links_chunks:
                                for l in links:
                                    text_results, _ = get_relevant_text_from_webpage(l)
                                    filtered_results = filter_data(text, text_results)
                                    if filtered_results:
                                        new_links.append(l)
                                        res.append(filtered_results)
                                        found = True
                                if found:
                                    break

                            f.write(json.dumps({'idx': i + 1, 'url': new_links, 'text': text, 'res': res}) + '\n')
                        else:
                            f.write(json.dumps({'idx': i + 1, 'url': d.get('url', []), 'text': text, 'res': new_r}) + '\n')
                except:
                    f.write(json.dumps({'idx': i + 1, 'text': text, 'res': []}) + '\n')
            else:
                f.write(json.dumps({'idx': i + 1, 'url': d['url'], 'text': text, 'res': d['res']}) + '\n')
    f.close()

def re_search():
    data = read_data()
    f = open('data1.json', 'a')
    for i, d in tqdm(enumerate(data)):
        text = d['text']
        if i >= 0:
            try:
                res = []
                if 'https://' not in text:
                    total_links = get_top_k_results_from_google(text, k=9)
                    links_chunks = list(divide_chunks(total_links, 3))

                    found = False
                    new_links = []
                    for links in links_chunks:
                        for l in links:
                            text_results, _ = get_relevant_text_from_webpage(l)
                            filtered_results = filter_data(text, text_results)
                            if filtered_results:
                                new_links.append(l)
                                res.append(filtered_results)
                                found = True
                        if found:
                            break

                    f.write(json.dumps({'idx': i+1, 'url': new_links, 'text': text, 'res': res}) + '\n')
                else:
                    new_r = []
                    for r in d['res']:
                        if r:
                            temp = []
                            for ir in r:
                                if 'JavaScript is disabled in this browser' in ir[1]:
                                    temp = []
                                    # new_r.append(r)
                                    break
                                else:
                                    temp.append(ir)
                            new_r.append(temp)
                        else:
                            new_r.append(r)
                    found = False
                    for r in new_r:
                        if r:
                            found = True
                            break
                    if not found:
                        total_links = get_top_k_results_from_google(text, k=9)
                        links_chunks = list(divide_chunks(total_links, 3))

                        found = False
                        new_links = []
                        for links in links_chunks:
                            for l in links:
                                text_results, _ = get_relevant_text_from_webpage(l)
                                filtered_results = filter_data(text, text_results)
                                if filtered_results:
                                    new_links.append(l)
                                    res.append(filtered_results)
                                    found = True
                            if found:
                                break

                        f.write(json.dumps({'idx': i+1, 'url': new_links, 'text': text, 'res': res}) + '\n')
                    else:
                        f.write(json.dumps({'idx': i + 1, 'url': d.get('url', []), 'text': text, 'res': new_r}) + '\n')
            except:
                f.write(json.dumps({'idx': i+1, 'text': text, 'res': []}) + '\n')


# refind()
# re_search()