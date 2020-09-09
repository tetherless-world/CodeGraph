import ijson
from bs4 import BeautifulSoup
import sys
import json
import re
from random import randrange, random

question_ids = set()
linked_ids = set()
unlinked_ids = set()
test_set = []

printed_ids = set()

first = True

def get_id(post):
  if 'id:' in post:
      return int(post['id:'])
  else:
      return int(post['id'])

  
def get_ids(post, postHtml):
    question_ids.add(get_id(post))


matchString = 'stackoverflow[.]com[/]questions[/](\d+)[/]'
pattern = re.compile(matchString)

def process(post, postHtml, ds):
    global first
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
                    sid = get_id(post)
                    linked_ids.add(int(sid))
                    linked_ids.add(int(tid))
                    test_set.append( (sid, tid, True) )
        except:
            pass


    if len(links) > 0:
        printed_ids.add(sid)
        
        newPostHtml = soup.get_text()
            
        out = { "id": sid, "text": newPostHtml, "related": links }

        if first:
            first = False
        else:
            ds.write(',')
        ds.write(json.dumps(out, sort_keys=True, indent=4))
    


def other_ids(post, postHtml, ds):
    global first
    id = get_id(post)
    if (id in unlinked_ids) or (id in linked_ids and not id in printed_ids):
            
        out = { "id": id, "text": postHtml, "related": [] }

        if first:
            first = False
        else:
            ds.write(',')
        ds.write(json.dumps(out, sort_keys=True, indent=4))

            
def dataset(postsPath, process):
    with open(postsPath, 'r') as posts:
        for post in ijson.items(posts, "item"):
            postHtml = post['text:']

            process(post, postHtml)

            for answer in post['answers']:
                process(answer, answer['text'])
                

if __name__ == '__main__':
    with open(sys.argv[2], 'w') as ds:
        ds.write("[")
        dataset(sys.argv[1], get_ids)
        dataset(sys.argv[1], lambda x, y: process(x, y, ds))

        falses = []
        unrelated = list(question_ids.difference(linked_ids))
        for x in test_set:
            if random() > .5:
                src = unrelated[randrange(0, len(unrelated), 1)]
                falses.append((src, x[1], False))
                unlinked_ids.add(src)
            else:
                dst = unrelated[randrange(0, len(unrelated), 1)]
                falses.append((x[0], dst, False))
                unlinked_ids.add(dst)

        test_set.extend(falses)
                            
        dataset(sys.argv[1], lambda x, y: other_ids(x, y, ds))
        ds.write("]")

    with open(sys.argv[3], 'w') as ts:
        ts.write(json.dumps(test_set, indent=4))
