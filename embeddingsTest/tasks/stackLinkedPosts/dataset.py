import ijson
from bs4 import BeautifulSoup
import sys
import json
import re

question_ids = []

def get_ids(post, postHtml):
    if ('id:' in post):
        question_ids.append(post['id:'])


matchString = 'stackoverflow[.]com[/]questions[/](\d+)[/]'
pattern = re.compile(matchString)

def process(post, postHtml):
    links = []
            
    soup = BeautifulSoup(postHtml, 'html.parser')
    for a in soup.find_all('a'):
        link = a.get( 'href' )
        url = pattern.search(link)
        if url is not None:
            id = url.group(1)
            if (id in question_ids):
                links.append(link)
                a.decompose()
            
    newPostHtml = soup.get_text()
            
    out = { "text": newPostHtml, "related": links }
    
    print(json.dumps(out, sort_keys=True, indent=4))

    
def dataset(postsPath, process):
    print("[")
    with open(postsPath, 'r') as posts:
        for post in ijson.items(posts, "item"):
            postHtml = post['text:']

            process(post, postHtml)

            for answer in post['answers']:
                process(answer, answer['text'])
                
    print("]")

if __name__ == '__main__':
    dataset(sys.argv[1], get_ids)
    dataset(sys.argv[1], process)
    print(question_ids)
