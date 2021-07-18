import json
import re


def read_data():
    data = []
    cnt = 1
    with open('val_links1.json') as f:
        for l in f:
            data.append(json.loads(l.strip()))
            print(cnt)
            cnt +=1
    return data


def prepare_summary():
    data = read_data()
    c = 1
    f = open('val_summ1_we.json', 'w')
    for d in data:
        summ = []
        for link in d['links']:
            if type(link[1]) == list:
                for text in link[1]:
                    if type(link[1]) == list:
                        summ.append(text[1])
                    else:
                        summ.append(text)
            elif type(link[1]) == str:
                summ.append(link[1])

                # print(link)
        claim = d['text']
        urls = re.findall(r'https?:\/+\/+t+\.+co+\/+\S*', d['text'])
        # new_links = []
        for li in urls:
            claim = claim.replace(li, '')
        claim=claim.strip()

        if summ:
            # d['summary'] = 'Claim= {} Text= {}'.format(claim, ' '.join(summ).replace('\n', '').replace('\t', ''))
            d['summary'] = '{}'.format(' '.join(summ).replace('\n', '').replace('\t', ''))
        else:
            d['summary'] = ''
            # c+=1
        f.write(json.dumps(d) + '\n')
        print()
    print(c)


if __name__ == '__main__':
    prepare_summary()
