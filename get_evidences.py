import time
import urllib

from googlesearch import search
import requests
from bs4 import BeautifulSoup


def get_top_k_results_from_google(text, k):
    text = urllib.parse.quote_plus(text)
    res = search(text)
    res = [r for r in res if '.pdf' not in r and 'twitter.com' not in r and 'facebook.com' not in r]
    if res:
        return res[:k]
    else:
        return []


def get_relevant_text_from_webpage(url_link):
    request = requests.get(url_link, verify=False, timeout=20)
    # time.sleep(1)
    Soup = BeautifulSoup(request.text, 'lxml')
    if 'hindi news' in Soup.text.lower() or 'hindi samachar' in Soup.text.lower() or '/download' in request.url :
        return [], request.url

    if '.xlsx' in request.url or '.txt' in request.url:
        return [], request.url

    if '.pdf' in request.url.lower():
        return [], request.url

    if '%PDF-' in request.text:
        return [], request.url

    # creating a list of all common heading tags
    heading_tags = ['h{}'.format(h) for h in range(1, 10)] + ['p']
    results = []

    for tags in Soup.find_all(heading_tags):
        if 'h' in tags.name:
            tokens = tags.text.strip().split()
            if len(tokens) > 8:
                results.append([tags.name, tags.text.strip()])
        else:
            tokens = tags.text.strip().split()
            if len(tokens) > 20:
                results.append([tags.name, tags.text.strip()])
    return results, request.url

if __name__ == '__main__':
    # q = 'Politically Correct Woman (Almost) Uses Pandemic as Excuse Not to Reuse Plastic Bag https://t.co/thF8GuNFPe #coronavirus #nashville'
    links = get_top_k_results_from_google("jhukh", k=3)
    print(links)
    for l in links:
        print(get_relevant_text_from_webpage(l))

    # url = 'https://twitter.com/TheOnion'
    # data = requests.get(url)
    # print(data.text)
