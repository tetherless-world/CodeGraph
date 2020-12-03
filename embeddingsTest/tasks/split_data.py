
import json
from shutil import copyfile
import random

def sample_SO_qa(infile, out_dir, base_name):
    all_qs = json.load(open(infile))
    random.shuffle(all_qs)
    num_training_qs = int(0.1 * len(all_qs))
    training_qs = []
    # with open(out_dir + 'stackoverflow_data_ranking_train.json', 'w', encoding='utf-8') as output_file:
    for q in all_qs[:num_training_qs]:
        training_qs.append(q)
    with open(out_dir + base_name + '_train.json', 'w', encoding='utf-8') as output_file:
        json.dump(training_qs, output_file, indent=2)
    testing_qs = []
    for q in all_qs[num_training_qs:]:
        testing_qs.append(q)
    with open(out_dir + base_name + '_testing.json', 'w', encoding='utf-8') as output_file:
        json.dump(testing_qs, output_file, indent=2)

    print('-'*10 + base_name +'-'*10)
    print("Total number of questions: ", len(all_qs))
    print("Total number of training questions: ", len(training_qs))
    print("Total number of testing questions: ", len(testing_qs))
    print('-'*20)

if __name__ == "__main__":
    # base_dir = '/home/ibrahim/CodeGraph/embeddingsTest/tasks/test_data'
    base_dir = '/home/ibrahim/blanca/'
    sample_SO_qa(base_dir + 'stackoverflow_data_ranking.json',
                 base_dir, 'stackoverflow_data_ranking')

    # sample_LinkedPost_qa('$DATA/stackoverflow_data_ranking.json', '')
    sample_SO_qa(base_dir + 'stackoverflow_matches_codesearchnet_5k.json',
                 base_dir, 'stackoverflow_matches_codesearchnet_5k')

    sample_SO_qa(base_dir + 'stackoverflow_matches_codesearchnet_5k_content.json',
                 base_dir, 'stackoverflow_matches_codesearchnet_5k_content')
