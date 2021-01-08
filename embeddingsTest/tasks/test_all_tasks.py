from multi_task_train import *
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
import argparse
import os
from utils import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_dir", help="data directory where all the files exist", required=True)
    parser.add_argument("--embed_type", help="embed type", required=True)
    parser.add_argument('--model_dir', help="model dir", required=False)
    parser.add_argument('--output_dir', help="output dir", required=False)
    args = parser.parse_args()

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################
    data_dir = args.data_dir

    if args.model_dir:
        model = util.get_model(args.embed_type, local_model_path=args.model_dir)
    else:
        model = util.get_model(args.embed_type)
        
    test_hierarchy_samples = create_hirerachy_examples(os.join(data_dir, 'hierarchy_test.json'))
    test_linked_posts = create_linked_posts(os.join(data_dir, 'stackoverflow_data_linkedposts__testing.json'))
    test_class_posts = create_train_class_posts(os.join(data_dir, 'class_posts_test_data.json'))
    test_usage = create_train_usage(os.join(data_dir, 'usage_test.json'))
    test_posts_ranking = create_posts_ranking(os.join(data_dir, 'stackoverflow_data_ranking_v2_testing.json'))
    test_search = create_search(os.join(data_dir, 'stackoverflow_matches_codesearchnet_5k_test_collection.tsv'),
                                 os.join(data_dir, 'stackoverflow_matches_codesearchnet_5k_test_queries.tsv'),
                                 os.join(data_dir, 'stackoverflow_matches_codesearchnet_5k_test_blanca-qidpidtriples.train.tsv'))


    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_hierarchy_samples, name='test-hierarchy-samples')
    test_evaluator(model, output_path=args.output_dir)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_posts_ranking, name='test-post-ranking')
    test_evaluator(model, output_path=args.output_dir)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_usage, name='test-usage')
    test_evaluator(model, output_path=args.output_dir)

    test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_linked_posts, name='test-linked-posts')
    test_evaluator(model, output_path=args.output_dir)
    test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_class_posts, name='test-class-posts')
    test_evaluator(model, output_path=args.output_dir)
    test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_search, name='test-search')
    test_evaluator(model, output_path=args.output_dir)
