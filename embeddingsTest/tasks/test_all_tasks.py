from multi_task_train import *
import sys

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################
test_hierarchy_samples = create_hirerachy_examples('/data/blanca/hierarchy_test.json')
test_linked_posts = create_linked_posts('/data/blanca/stackoverflow_data_linkedposts__testing.json')
test_class_posts = create_train_class_posts('/data/blanca/class_posts_test_data.json')
test_usage = create_train_usage('/data/blanca/usage_test.json')
test_posts_ranking = create_posts_ranking('/data/blanca/stackoverflow_data_ranking_v2_testing.json')
test_search = create_search('/data/blanca/stackoverflow_matches_codesearchnet_5k_test_collection.tsv',
                             '/data/blanca/stackoverflow_matches_codesearchnet_5k_test_queries.tsv',
                             '/data/blanca/stackoverflow_matches_codesearchnet_5k_test_blanca-qidpidtriples.train.tsv')

model_save_path = sys.argv[1]
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_hierarchy_samples, name='test-hierarchy-samples')
test_evaluator(model, output_path=model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_posts_ranking, name='test-post-ranking')
test_evaluator(model, output_path=model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_usage, name='test-usage')
test_evaluator(model, output_path=model_save_path)

test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_linked_posts, name='test-linked-posts')
test_evaluator(model, output_path=model_save_path)
test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_class_posts, name='test-class-posts')
test_evaluator(model, output_path=model_save_path)
test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_search, name='test-search')
test_evaluator(model, output_path=model_save_path)
