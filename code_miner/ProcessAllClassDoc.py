import re
import json
import MongoDBHandler

"""
This code takes a local mongo db handler and gets all the class usage documentation, and dumps into a 
format that is processable by the BERT QA interface.  Note that the process of using BERT works best in 
terms of batches, so its better to process all class documentation in a batch rather than iteratively.  The class
MachineComprehensionInterface can be used for individual calls but the overhead is significant.
"""
def process_all_class_docs():
    mongo = MongoDBHandler.MongoDBHandler()
    all_classes = mongo.get_distinct_classes('sklearn')
    qa = {}
    docs = []
    qa['data'] = docs
    qis = 0
    all_doc = {}

    for principal_class in all_classes:

        snippets = mongo.get_class_usage_doc('sklearn', principal_class)

        if snippets is None:
            continue

        for snippet in snippets:
            if snippet['title'] in all_doc:
                continue
            all_doc[snippet['title']] = 1
            classes = set(re.findall(':class:`([^`]*)`', snippet['description']))
            questions = []
            paragraphs = []

            for c in classes:
                # if there are classes mentioned that are not the principal class, then this is likely a comparison with
                # another class
                if c != principal_class:
                    questions.append({'question': 'What is ' + c + ' useful for?', 'id': qis})
                    qis += 1
                    questions.append({'question': 'How is ' + c + ' different from ' + principal_class + '?', 'id': qis})
                    qis += 1
                else:
                    if len(classes) == 1:
                        questions.append({'question': 'What is ' + c + ' useful for?', 'id': qis})
                        qis += 1
            questions.append({'question': 'What are the important features of ' + principal_class + '?', 'id': qis})
            qis += 1

            paragraph = {'context': snippet['description'], 'qas': questions}
            paragraphs.append(paragraph)
            docs.append({'title': snippet['title'], 'paragraphs': paragraphs})

        # add class doc in
        class_doc = mongo.get_class_doc('sklearn', principal_class)
        questions = []
        questions.append({'question': 'What are the important features of ' + principal_class + '?', 'id': qis})
        qis += 1
        paragraph = {'context': class_doc, 'qas': questions}
        paragraphs.append(paragraph)
        docs.append({'title': principal_class, 'paragraphs': paragraphs})

    with open('../bert_results/all_classes_classdoc.json', 'w') as f:
        f.write(json.dumps(qa, indent=4))


def main():
    process_all_class_docs()


if __name__ == "__main__":
    main()
