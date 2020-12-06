
import json, re
from shutil import copyfile
import random
from bs4 import BeautifulSoup


def prepare_ranking_data(all_qs):
    training_qs = []
    # with open(out_dir + 'stackoverflow_data_ranking_train.json', 'w', encoding='utf-8') as output_file:
    for q in all_qs:
        q_id = q['id:']
        q_text = q['title'] + ' ' + q['text:']
        q_url = q['url']
        # accepted_ans_id = q['AcceptedAnswerId']
        q_votes = q['votes:']
        q_data = {}
        q_data['q_id'] = q_id
        q_data['q_url'] = q_url
        q_data['q_text'] = q_text
        q_data['q_votes'] = q_votes

        votes_2_text = {}
        for ans in q['answers']:
            if ans['votes']:
                if int(ans['votes']) not in votes_2_text:
                    # print('two answers with the same vote??')
                    votes_2_text[int(ans['votes'])] = []
                votes_2_text[int(ans['votes'])].append((ans['text'], ans['accepted']))
        rank = 1
        for a_vote in sorted(votes_2_text.keys(), reverse=True):
            for (a_text, a_accepted) in votes_2_text[a_vote]:
                # a_text, a_accepted = votes_2_text[a_vote]
                a_data = dict(q_data)
                a_data['a_text'] = a_text
                a_data['a_accepted'] = a_accepted
                a_data['a_votes'] = a_vote
                a_data['a_rank'] = rank
                rank += 1
                training_qs.append(a_data)
    return training_qs

def preparse_search_data(all_qs):
    training_qs = []
    # with open(out_dir + 'stackoverflow_data_ranking_train.json', 'w', encoding='utf-8') as output_file:
    for q in all_qs:
        rank = 1
        for idx, match in enumerate(q['matches']):
            q_data = {}
            q_data['query'] = q['query']
            q_data['q_id'] = match['question_id']
            q_data['q_url'] = "https://stackoverflow.com/questions/"+match['question_id']
            q_data['q_title'] =  match['question_title']
            q_data['q_text'] = match['question_text']
            q_data['a_text'] = ''
            for ans in match['answers']:
                q_data['a_text'] = ans['answer_text'] + ' '
            q_data['rank'] = rank
            rank += 1
            training_qs.append(q_data)
    return training_qs

def preparse_linkedPost(all_qs, url2post, all_urls, num_neg = 3):
    training_qs = []
    # with open(out_dir + 'stackoverflow_data_ranking_train.json', 'w', encoding='utf-8') as output_file:
    for q in all_qs:
        q_id = q['id:']
        q_text = q['title'] + ' ' + q['text:']
        q_url = q['url']
        # accepted_ans_id = q['AcceptedAnswerId']
        # q_votes = q['votes:']
        q_data = {}
        # q_data['q_id'] = q_id
        q_data['url_1'] = q_url

        a_text = ''
        for ans in q['answers']:
            a_text = ans['text'] + ' '
        q_data['text_1'] = q_text + ' ' + a_text
        links = extract_links(q_data['text_1'])
        found_pos = False
        for link in links:
            if link == q_url:
                continue

            to_add = []
            new_data = dict(q_data)
            if link in url2post:
                url2, text2 = url2post[link]
                new_data['url_2'] = url2
                new_data['text_2'] = text2
                new_data['class'] = 'relevant'
                to_add.append(new_data)
                found_pos = True
            if found_pos:
                for i in range(num_neg):
                    rand_choice = random.choice(all_urls)
                    while rand_choice in links or rand_choice == q_url:
                        rand_choice = random.choice(all_urls)
                    new_data = dict(q_data)
                    url2, text2 = url2post[rand_choice]
                    new_data['url_2'] = url2
                    new_data['text_2'] = text2
                    new_data['class'] = 'irrelevant'
                    to_add.append(new_data)
            training_qs.extend(to_add)

    return training_qs


def sample_SO_qa(infile, out_dir, base_name, search_task = False):
    all_qs = json.load(open(infile))
    random.shuffle(all_qs)
    num_training_qs = int(0.1 * len(all_qs))
    if search_task:
        training_qs = preparse_search_data(all_qs[:num_training_qs])
    else:
        training_qs = prepare_ranking_data(all_qs[:num_training_qs])
    with open(out_dir + base_name + '_train.json', 'w', encoding='utf-8') as output_file:
        json.dump(training_qs, output_file, indent=2)


    if search_task:
        testing_qs = preparse_search_data(all_qs[num_training_qs:])
    else:
        testing_qs = prepare_ranking_data(all_qs[num_training_qs:])
    with open(out_dir + base_name + '_testing.json', 'w', encoding='utf-8') as output_file:
        json.dump(testing_qs, output_file, indent=2)

    print('-'*10 + base_name +'-'*10)
    print("Total number of questions: ", len(all_qs))
    print("Total number of training questions: ", len(training_qs))
    print("Total number of testing questions: ", len(testing_qs))
    print('-'*20)



def sample_linked_qa(infile, out_dir, base_name):

    all_qs = json.load(open(infile))
    url2post = {}
    all_urls = []
    for q in all_qs:
        q_id = q['id:']
        q_text = q['title'] + ' ' + q['text:']
        q_url = q['url']
        url2post[q_url] = (q_url, q_text)
        all_urls.append(q_url)
    random.shuffle(all_qs)
    num_training_qs = int(0.1 * len(all_qs))



    training_qs = preparse_linkedPost(all_qs[:num_training_qs], url2post, all_urls)
    with open(out_dir + base_name + '_train.json', 'w', encoding='utf-8') as output_file:
        json.dump(training_qs, output_file, indent=2)

    testing_qs = preparse_linkedPost(all_qs[num_training_qs:], url2post, all_urls)
    with open(out_dir + base_name + '_testing.json', 'w', encoding='utf-8') as output_file:
        json.dump(testing_qs, output_file, indent=2)

    print('-'*10 + base_name +'-'*10)
    print("Total number of questions: ", len(all_qs))
    print("Total number of training questions: ", len(training_qs))
    print("Total number of testing questions: ", len(testing_qs))
    print('-'*20)

def extract_links(postHtml):
    matchString = 'stackoverflow[.]com[/]questions[/](\d+)[/]'
    pattern = re.compile(matchString)

    links = []

    soup = BeautifulSoup(postHtml, 'html.parser')
    for a in soup.find_all('a'):
        try:
            link = a.get('href')
            url = pattern.search(link)
            if url is not None:
                tid = int(url.group(1))
                # if (tid in question_ids):

                link = 'https://stackoverflow.com/questions/'+str(tid)
                links.append(link)
                    # a.decompose()
                    # sid = get_id(post)
                    # linked_ids.add(int(sid))
                    # linked_ids.add(int(tid))
                    # test_set.append((sid, tid, True))
        except:
            pass
    return links



if __name__ == "__main__":
    base_dir = '/Users/ibrahimabdelaziz/ibm/github/CodeGraph/embeddingsTest/tasks/test_data/'
    # base_dir = '/data/blanca/'
    # sample_SO_qa(base_dir + 'stackoverflow_data_ranking.json',
    #              base_dir, 'stackoverflow_data_ranking')

    sample_linked_qa(base_dir + 'stackoverflow_data_ranking.json',
                 base_dir, 'stackoverflow_data_linkedposts_')

    # # sample_LinkedPost_qa('$DATA/stackoverflow_data_ranking.json', '')
    # sample_SO_qa(base_dir + 'stackoverflow_matches_codesearchnet_5k.json',
    #              base_dir, 'stackoverflow_matches_codesearchnet_5k', search_task=True)
    # #
    # sample_SO_qa(base_dir + 'stackoverflow_matches_codesearchnet_5k_content.json',
    #              base_dir, 'stackoverflow_matches_codesearchnet_5k_content', search_task=True)
