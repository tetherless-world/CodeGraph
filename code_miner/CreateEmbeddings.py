import MongoDBHandler
import faiss
import sys

from bert_serving.client import BertClient

class EmbeddingsCreator(object):

    def __init__(self):
        self.bc = BertClient(ip='gputest2.sl.cloud9.ibm.com')

    def fetch_relevant_text(self):
        mongo = MongoDBHandler.MongoDBHandler()
        class_keys = {}
        method_keys = {}
        all_class_sentences = []
        all_method_sentences = []

        for d in mongo.db['sklearn'].find():
            if d['class_name'] not in class_keys:
                if d['class_doc'] is not None:
                    all_class_sentences.append(d['class_name'] + "\n" + d['class_doc'])
                    class_keys[d['class_name']] = 1
                if d['class_name'] + d['method'] not in method_keys:
                    if d['overall_doc'] is not None:
                        all_method_sentences.append(d['class_name'] + "\n" + d['method'] + "\n" + d['overall_doc'])
                        method_keys[d['class_name'] + d['method']] = 1

        print(len(method_keys))
        with open('/tmp/methods', 'w') as out:
            for m in method_keys:
                out.write(m + '\n')
        with open('/tmp/classes', 'w') as out:
            for m in class_keys:
                out.write(m + '\n')
        with open('/tmp/classes_txt', 'w') as out:
            for m in all_class_sentences:
                out.write(m + '\n')
        with open('/tmp/methods_txt', 'w') as out:
            for m in all_method_sentences:
                out.write(m + '\n')
        return all_class_sentences, all_method_sentences

    def extract_embeddings(self, sentences, embeddings_file_name, index_name):
        embeddings = self.bc.encode(sentences)
        embeddings.dump(embeddings_file_name)
        print(embeddings.shape)
        print(embeddings)
        index = faiss.IndexFlatL2(768)
        index.add(embeddings)
        faiss.write_index(index, index_name)

    def creaate_class_and_method_embeddings(self):
        all_class_sentences, all_method_sentences = self.fetch_relevant_text()
        # pointer to where bert is hosted as as service
        bc = BertClient(ip='gputest2.sl.cloud9.ibm.com')
        self.extract_embeddings(all_class_sentences, '/tmp/class_embedding_matrix', '/tmp/class_index')
        self.extract_embeddings(all_method_sentences, '/tmp/method_embedding_matrix', '/tmp/method_index')

    def create_query_embeddings(self, all_queries, embeddings_file):
        embeddings = self.bc.encode(all_queries)
        embeddings.dump(embeddings_file)


def main():
    embed = EmbeddingsCreator()
    if len(sys.argv) > 1:
        queries_file = sys.argv[1]
        embeddings_file = sys.argv[2]
        with open(queries_file, 'r') as f:
            allqueries = f.readlines()
            # debugging purposes only
            for index, q in enumerate(allqueries):
                if '|' in q:
                    res = q.split('|')
                    st = '\n'.join(res)
                    print(st)
                    allqueries[index] = st

        if len(sys.argv) == 3:
            embed.create_query_embeddings(allqueries, embeddings_file)
        else:
            queries_index = sys.argv[3]
            embed.extract_embeddings(allqueries, embeddings_file, queries_index)

    else:
        embed.creaate_class_and_method_embeddings()


if __name__ == '__main__':
    main()
