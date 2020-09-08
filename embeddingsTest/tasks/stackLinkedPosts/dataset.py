import ijson
from bs4 import BeautifulSoup
import sys
import json
import re
from random import randrange

question_ids = set()
linked_ids = set()
test_set = []

def get_ids(post, postHtml):
    if ('id:' in post):
        question_ids.add(int(post['id:']))


matchString = 'stackoverflow[.]com[/]questions[/](\d+)[/]'
pattern = re.compile(matchString)

def process(post, postHtml):
    links = []
            
    soup = BeautifulSoup(postHtml, 'html.parser')
    for a in soup.find_all('a'):
        try:
            link = a.get( 'href' )
            url = pattern.search(link)
            if url is not None:
                tid = int(url.group(1))
                if (tid in question_ids):
                    links.append(link)
                    a.decompose()
                    print("ADDING " + str(tid))
                    if 'id:' in post:
                        sid = int(post['id:'])
                    else:
                        sid = int(post['id'])
                    linked_ids.add(int(sid))
                    linked_ids.add(int(tid))
                    print("ADDING " + str( (sid, tid, True) ))
                    test_set.append( (sid, tid, True) )
        except:
            pass
            
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

    print(linked_ids)
    unrelated = list(question_ids.difference(linked_ids))
    for id in linked_ids:
        test_set.append((id, unrelated[randrange(0, len(unrelated), 1)], False))
        test_set.append((unrelated[randrange(0, len(unrelated), 1)], id, False))

    print(test_set)
    print(question_ids)
    print(linked_ids)
