
import json, re, sys, time, os
from shutil import copyfile
import random
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch


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


def do_boolean_and_query(key_terms=None):
    must_clauses = []

    for term in key_terms.split(' '):
        must_clauses.append({"match": {"content": term}})
    query = {
        "from": 0, "size": 200,
        "query": {
            "bool": {
                "must": []
            }
        }
    }
    query['query']['bool']['must'] = must_clauses
    # query = {
    #     "from": 0, "size": 100,
    #     "query": {
    #         "query_string": {
    #             "query": "FixedLenFeature",
    #             "default_field": "content"
    #         }
    #     }
    # }

    return query

def extract_class_mentions(output_dir, classes_map_file):
    file = open(classes_map_file, 'r')
    class_list = {}
    for line in file:
        # if 'RandomForestClassifier' not in line:  #debug
        #     continue
        names = line.strip().split(' ')
        # parts = names[0].split('.')
        parts =  re.split("\.", names[0])
        key = parts[0] + ' ' + parts[-1]
        class_list[key] = [names]
    matches = []
    es = Elasticsearch([{'host': 'localhost','port': 9200}])
    for idx, short_class_name in enumerate(class_list):
        parts = short_class_name.split(' ')
        package, class_name = parts[0], parts[1]
        # res = es.search(index=sys.argv[1], body=get_pure_class_or_function_query(sys.argv[2]))
        query = do_boolean_and_query(class_name)
        start = time.time()
        res = es.search(index='stackoverflow3', body=query)
        print('Search for class: ', short_class_name, '({} out of {}'.format(idx, len(class_list)),
              ', num of results = ', len(res['hits']['hits']), ', took ', (time.time()-start), 'sec')
        # stack_answers = []
        num_matches = 0
        for qa in res['hits']['hits']:
            stack_answer = {}
            stack_answer['id'] = qa['_source']['question_id:']
            stack_answer['title'] = qa['_source']['title']
            stack_answer['text'] = qa['_source']['question_text:']
            answers = []
            answers_text = ''
            for ans in qa['_source']['answers']:
                # aId, aPostTypeId, aParentId, aAcceptedAnswerId, answerTitle, answerBody, aTags, avotes = answer
                answer = {}
                answer['answer_id'] = ans[0]
                answer['answer_text'] = ans[5]
                answers_text += ans[5] + ' '
                answer['answer_votes'] = ans[7]
                answers.append(answer)

            stack_answer['answers'] = answers

            submodule_names = set([])
            for aliases in class_list[short_class_name]:
                for alias in aliases:
                    # parts = alias.split('\.')[0]   #[:-1] #focus only on package name only
                    parts = re.split("\.", alias)[0]
                    if parts != class_name:
                        submodule_names.add(parts)
            cond1_submodule_in_text = False
            for submodule in submodule_names:
                regex_submodule_name = r"(\b{}\b)".format(submodule)
                if re.search(regex_submodule_name, qa['_source']['content']):
                    cond1_submodule_in_text = True
                    break
            cond2_package_in_q_a = False

            regex_class_name = r"(\b{}\b)".format(class_name)
            # matches = re.finditer(regex_class_name, qa['_source']['question_text:'])
            # for matchNum, match in enumerate(matches, start=1):
            #     print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum,
            #                                                                         start=match.start(),
            #                                                                         end=match.end(),
            #                                                                         match=match.group()))


            # if re.search(regex_class_name, qa['_source']['question_text:']) and re.search(regex_class_name, answers_text):
            if re.search(regex_class_name, qa['_source']['title']) and re.search(regex_class_name, answers_text):
                cond2_package_in_q_a = True
            # for short_class_name, full_names in class_list.items():
            #     parts = short_class_name.split(' ')
            # if re.search(r'\b'+class_name+'\b', qa['_source']['title']) or \
            #         ( re.search(r'\b'+package+'\b', qa['_source']['content']) and re.search(r'\b'+class_name+'\b', qa['_source']['content'])):
            # # if class_name in qa['_source']['title'] or (package in qa['_source']['content'] and class_name in qa['_source']['content']):
            if cond2_package_in_q_a and cond1_submodule_in_text:
                q_info_cp = dict(stack_answer)
                q_info_cp['relevant_class'] = class_name
                q_info_cp['relevant_class_alias'] = class_list[short_class_name]
                matches.append(q_info_cp)
                if num_matches > 5:
                    break
                num_matches += 1
        print('# of matches after filtering: ', num_matches)
        # stack_answers.append(stack_answer)
        if len(matches) % 1000 == 0:
            print('Saving intermediate matches, len = ', len(matches))
            with open(output_dir + 'class_matches_in_stackoverflow_v4.json', 'w', encoding='utf-8') as output_file:
                json.dump(matches, output_file, indent=2)
        print('Total time filtering: ', time.time() - start)
    with open(output_dir + 'class_matches_in_stackoverflow_v4.json', 'w', encoding='utf-8') as output_file:
        json.dump(matches, output_file, indent=2)
    random.shuffle(matches)
    with open(output_dir + 'sample_class_matches_in_stackoverflow_v4.json', 'w', encoding='utf-8') as output_file:
        json.dump(matches[:100], output_file, indent=2)

    print('Done -- saved {} matches in total'.format(len(matches)))

def check_docstr_intersection(docstring_dir, forum_2_class_file):
    # out_triple_file = open(out_dir+'/docstrings_triples.nq', 'w')
    all_klasses_found = set([])
    klass_2_docstr = {}
    for lib in os.listdir(docstring_dir):
        if lib.startswith('.'):
            print('Skip ', lib)
            continue
        source_path = os.path.join(docstring_dir, lib)
        if not os.path.isdir(source_path):  # '../data/mods.22/pyvenv.cfg'
            continue
        for f in os.listdir(source_path):
            # if len(all_klasses_found) > 10000:
            #     break
            if not f.endswith('.json'):
                print('Skip ', lib)
                continue
            pth = os.path.join(source_path, f)
            with open(pth) as input:
                try:
                    functions = json.load(input)
                except:
                    print('Exception during loading file:' + pth)
                    continue
                if type(functions) == dict:
                    functions = [functions]
                for function_dic in functions:
                    if 'klass' in function_dic and 'class_docstring' in function_dic and function_dic['class_docstring'].strip() != '':
                        all_klasses_found.add(function_dic['klass'])
    all_qs = json.load(open(forum_2_class_file))
    num_intersectiion = 0
    for match in all_qs:
        for alias in match['relevant_class_alias'][0]:
            if alias in all_klasses_found:
                num_intersectiion += 1
                break
    print('Total number of classes loaded = ', len(all_klasses_found))
    print('Number of forum to class matches = ', len(all_qs), ', intersection = ', num_intersectiion)

if __name__ == "__main__":
    # base_dir = '/Users/ibrahimabdelaziz/ibm/github/CodeGraph/embeddingsTest/tasks/test_data/'
    # base_dir = '/data/blanca/'
    # sample_SO_qa(base_dir + 'stackoverflow_data_ranking.json',
    #              base_dir, 'stackoverflow_data_ranking')

    # sample_linked_qa(base_dir + 'stackoverflow_data_ranking.json',
    #              base_dir, 'stackoverflow_data_linkedposts_')

    # # sample_LinkedPost_qa('$DATA/stackoverflow_data_ranking.json', '')
    # sample_SO_qa(base_dir + 'stackoverflow_matches_codesearchnet_5k.json',
    #              base_dir, 'stackoverflow_matches_codesearchnet_5k', search_task=True)
    # #
    # sample_SO_qa(base_dir + 'stackoverflow_matches_codesearchnet_5k_content.json',
    #              base_dir, 'stackoverflow_matches_codesearchnet_5k_content', search_task=True)


    output_dir = './test_data/'
    classes_map_file = './test_data/classes.map'
    extract_class_mentions(output_dir, classes_map_file)

    # check_docstr_intersection("/Users/ibrahimabdelaziz/ibm/github/code_knowledge_graph/data/mods.22/",
    #                           '/Users/ibrahimabdelaziz/Downloads/sample_class_matches_in_stackoverflow_v3.json')
    # check_docstr_intersection("/home/ibrahim/full_docstrings-merge-15-22/",
    #                           './test_data/class_matches_in_stackoverflow_v4.json')

    # all_qs = json.load(open('/Users/ibrahimabdelaziz/Downloads/sample_class_matches_in_stackoverflow_v3.json'))
    # resultfile = open("/Users/ibrahimabdelaziz/Downloads/manual_eval_sample_class_matches.csv", "w")
    #
    # for q in all_qs:
    #     # resultfile.write('{}\t{}\t{}\t{}\n'.format(q['relevant_class'], q['relevant_class_alias'], q['title'],'https://stackoverflow.com/questions/'+q['id']))
    #     resultfile.write('{},{},{}\n'.format(q['title'].replace(',', ' '), '-'.join(q['relevant_class_alias'][0]), 'https://stackoverflow.com/questions/'+q['id']))
    #
    # resultfile.close()

