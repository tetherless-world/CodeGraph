from sklearn import *
import sklearn.utils as utils
import rdflib
import re

ql = utils.testing.all_estimators(include_dont_test=True, include_meta_estimators=True, include_other=True)
nl = []
for cls in ql:
    print('"' + cls[0] + '." ')


# The following query does not scale!
"""
g = rdflib.Graph()

# ... add some triples to g somehow ...
g.parse("../data/rdfGraph.ttl", format="ttl")
for c in nl:
    name = c.__class__.__module__ + '.' + c.__class__.__name__
    sparql = 'SELECT ?z WHERE {?a <http://edge/dataflow> ?z . ?a <http://path> "' + name + '" .}'
    qres = g.query(sparql)

    for row in qres:
        print("%s" % row)
"""
