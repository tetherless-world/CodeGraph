import pymongo
from pymongo import MongoClient
from bson.son import SON

class MongoDBHandler(object):

    def __init__(self, host='localhost', port=27017):
        self.client = MongoClient(host, port)
        self.db = self.client.code_kg

    def insert_one(self, collection, doc):
        self.db[collection].insert_one(doc)

    def query_doc(self, collection, doc_selector):
        return self.db[collection].find(doc_selector)

    def insert_many(self, collection, docs):
        self.db[collection].insert_many(docs)

    ## This does not seem to do anything.  Need to set it from the mongo shell for now for some reason
    def create_text_index(self, collection):
        self.db.collection.create_index(
            [
                ('class_doc', 'text'),
                ('class_usage_doc', 'text'),
                ('method_doc', 'text'), ('overall_doc', 'text'),
                ('dso_keytext', 'text')
            ]
        )

    def count(self, collection):
        return self.db[collection].find().count()

    def count_unique_methods(self, collection):
        return len(self.find_unique_methods(collection))

    def find_unique_methods(self, collection):
        keys = []
        for d in self.db[collection].find():
            keys.append(d['class_name'] + d['method'])
        return set(keys)

    def get_distinct_classes(self, collection):
        return self.db[collection].distinct('class_name')

    def get_class_usage_doc(self, collection, clazz):
        query = {}
        query['class_name'] = clazz
        return self.db[collection].find_one(query)['class_usage_doc']

    def get_class_doc(self, collection, clazz):
        query = {}
        query['class_name'] = clazz
        return self.db[collection].find_one(query)['class_doc']

    def list_indexes(self, collection):
        return self.db[collection].getIndexes()

    def drop_db(self, collection):
        self.db[collection].drop


