import sys
import json
import argparse

def run_analysis(titles_file, content_file, top_k):
    num_matches = 0
    all_matches = 0
    total_search_rank = 0
    
    with open(titles_file) as f:
        search_data_titles = json.load(f)

        with open (content_file) as fi:
            search_data_content = json.load(fi)
        
            for index, obj in enumerate(search_data_content):
                search_matches = {}
                search_ranks = []
                if index == len(search_data_titles):
                    continue
                for m in search_data_titles[index]['matches']:
                    search_matches[m['question_id']] = 1
                    search_ranks.append(m['question_id'])
                    
                query = obj['query']

                for j, match in enumerate(obj['matches']):
                    qid = match['question_id']
                    if j > top_k:
                        break

                    if qid in search_matches:
                        num_matches += 1
                        total_search_rank += search_ranks.index(qid)
                        
                    all_matches += 1
    print("overlap in matches:" + str(num_matches/all_matches))
    print("average search rank:" + str(total_search_rank/num_matches))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an evaluation on two types of text search')
    parser.add_argument('--search_titles_file', type=str,
                        help='search results only for titles')
    parser.add_argument('--search_contents_file', type=str,
                        help='search results for contents + title')
    parser.add_argument('--top_k', type=int,
                        help='how many matches to consider in computing overlap')
    args = parser.parse_args()
    run_analysis(args.search_titles_file, args.search_contents_file, args.top_k)
