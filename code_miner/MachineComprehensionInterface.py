import run_squad
import tensorflow as tf
import modeling
import tokenization
import collections
import os
import json
import sys

flags = tf.flags

FLAGS = flags.FLAGS


class BERT_QA(object):

    def __init__(self):
        self.estimator = None
        self.tokenizer = None
        self.sess = tf.Session()

    ###
    # this code is unfortunately exactly BERT code but with the annoying aspect that it takes a json object
    # as input rather than a file
    def read_squad_examples(self, input_data, is_training):

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if is_training:

                        if FLAGS.version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length -
                                                               1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(
                                doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                tokenization.whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                                   actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = run_squad.SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)

        return examples

    def init_bert(self):
        bert_config = modeling.BertConfig.from_json_file(run_squad.FLAGS.bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=run_squad.FLAGS.vocab_file,
            do_lower_case=run_squad.FLAGS.do_lower_case)

        tf.logging.set_verbosity(tf.logging.DEBUG)

        num_train_steps = None
        num_warmup_steps = None

        model_fn = run_squad.model_fn_builder(
          bert_config=bert_config,
          init_checkpoint=run_squad.FLAGS.init_checkpoint,
          learning_rate=run_squad.FLAGS.learning_rate,
          num_train_steps=num_train_steps,
          num_warmup_steps=num_warmup_steps,
          use_tpu=run_squad.FLAGS.use_tpu,
          use_one_hot_embeddings=run_squad.FLAGS.use_tpu)

        tpu_cluster_resolver = None
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

        run_config = tf.contrib.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          master=run_squad.FLAGS.master,
          model_dir=run_squad.FLAGS.output_dir,
          save_checkpoints_steps=run_squad.FLAGS.save_checkpoints_steps,
          tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=run_squad.FLAGS.iterations_per_loop,
              num_shards=run_squad.FLAGS.num_tpu_cores,
              per_host_input_for_training=is_per_host))

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=False,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=run_squad.FLAGS.train_batch_size,
          predict_batch_size=run_squad.FLAGS.predict_batch_size)

    # Another unfortunate copy of the existing run_squad method just to pass output back as JSON rather than write it to
    # a file
    def write_predictions(self, all_examples, all_features, all_results, n_best_size,
                          max_answer_length, do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file):
        """Write final predictions to the json file and log-odds of null if needed."""
        tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
        tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min mull score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = run_squad._get_best_indexes(result.start_logits, n_best_size)
                end_indexes = run_squad._get_best_indexes(result.end_logits, n_best_size)
                # if we could have irrelevant answers, get the min score of irrelevant
                if FLAGS.version_2_with_negative:
                    feature_null_score = result.start_logits[0] + result.end_logits[0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = result.start_logits[0]
                        null_end_logit = result.end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index]))

            if FLAGS.version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = run_squad.get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit))

            # if we didn't inlude the empty option in the n-best, inlcude it
            if FLAGS.version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(
                        _NbestPrediction(
                            text="", start_logit=null_start_logit,
                            end_logit=null_end_logit))
            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = run_squad._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            if not FLAGS.version_2_with_negative:
                all_predictions[example.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score - the score of best non-null > threshold
                score_diff = score_null - best_non_null_entry.start_logit - (
                    best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > FLAGS.null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text

            all_nbest_json[example.qas_id] = nbest_json

        if FLAGS.version_2_with_negative:
            return all_predictions, all_nbest_json, scores_diff_json
        return all_predictions, all_nbest_json

    def do_predict(self, json_data):
        eval_examples = self.read_squad_examples(
            input_data=json_data, is_training=False)

        eval_writer = run_squad.FeatureWriter(
            filename=os.path.join(run_squad.FLAGS.output_dir, "eval.tf_record"),
            is_training=False)

        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        run_squad.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=self.tokenizer,
            max_seq_length=run_squad.FLAGS.max_seq_length,
            doc_stride=run_squad.FLAGS.doc_stride,
            max_query_length=run_squad.FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", run_squad.FLAGS.predict_batch_size)

        all_results = []

        predict_input_fn = run_squad.input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=run_squad.FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in self.estimator.predict(
            predict_input_fn, yield_single_examples=True):
          if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(all_results)))
          unique_id = int(result["unique_ids"])
          start_logits = [float(x) for x in result["start_logits"].flat]
          end_logits = [float(x) for x in result["end_logits"].flat]
          all_results.append(
              run_squad.RawResult(
                  unique_id=unique_id,
                  start_logits=start_logits,
                  end_logits=end_logits))

        output_prediction_file = os.path.join(run_squad.FLAGS.output_dir, "predictions.json")
        output_nbest_file = os.path.join(run_squad.FLAGS.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(run_squad.FLAGS.output_dir, "null_odds.json")

        return self.write_predictions(eval_examples, eval_features, all_results,
                          run_squad.FLAGS.n_best_size, run_squad.FLAGS.max_answer_length,
                          run_squad.FLAGS.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file)


def main():
    bert = BERT_QA()
    bert.init_bert()

    with tf.gfile.Open(sys.argv[1], "r") as reader:
        input_data = json.load(reader)["data"]
        allbest, nbest = bert.do_predict(input_data)

        items = list(allbest.items())
        qid2answer = dict(items)
        for doc in input_data:
            for q in doc['paragraphs'][0]['qas']:
                if q['id'] in allbest:
                    q['answer'] = qid2answer[q['id']]
        with open(os.path.join(sys.argv[2], 'bert_answers.txt'), 'w') as out:
            out.write(json.dumps(input_data, indent=4))


if __name__ == '__main__':
    main()