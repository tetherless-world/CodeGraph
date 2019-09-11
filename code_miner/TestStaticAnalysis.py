import unittest
import requests
import json
# import VisualizeStaticAnalysis
from code_miner import CollectAllGraphs
from code_miner import RDFGraphHandler
import rdflib
import rdflib.plugins.sparql as sparql

url = 'http://localhost:4567/analyze_code'
constructor_successors = "../static_analysis_queries/constructor_successors.sparql"
constructor_predecessors = "../static_analysis_queries/constructor_predecessors.sparql"
generic_successors = "../static_analysis_queries/generic_successors.sparql"
generic_predecessors = "../static_analysis_queries/generic_predecessors.sparql"

def get_graph(f):
    with open(f) as pf:

        source = '\n'.join(pf.readlines()).encode('UTF-8')
    res = requests.post(url=url,
                        data=source,
                        headers={'Content-Type': 'application/octet-stream'})
    assert res.text != '<html><body><h2>500 Internal Server Error</h2></body></html>'
    assert res.text != '[]'
    json_data = json.loads(res.text)
    print(json.dumps(json_data, indent=4))
    assert len(json_data) > 0
    nodes, data_flow_edges, control_flow_edges = VisualizeStaticAnalysis.parse_wala_into_graph(json_data, add_args=True)
    g = (nodes, data_flow_edges, control_flow_edges)
    rdf_graph = rdflib.Graph()
    CollectAllGraphs.addToGraph(rdf_graph, g)
    return rdf_graph

def get_graph_edges(g):
    triples = []
    for subject, predicate, obj in g:
        triples.append((subject, predicate, obj))
    return triples


"""
   Cant reuse methods in RDFHandler because they use prepared queries which do not seem to work against
   a dynamically created graph.  Most annoying
"""
def get_query(rdf_graph, method, query):
    with open(query, 'r') as f:
        query_str = f.read()
    query_str = query_str.replace('?m', method)
    results = rdf_graph.query(query_str)
    ret = []
    for row in results:
        ret.append(row[0].value)
    return ret

"""
   Cant reuse methods in RDFHandler because they use prepared queries which do not seem to work against
   a dynamically created graph.  Most annoying
"""
def get_query(rdf_graph, method, query):
    with open(query, 'r') as f:
        query_str = f.read()
    query_str = query_str.replace('?m', method)
    results = rdf_graph.query(query_str)
    ret = []
    for row in results:
        ret.append(row[0].value)
    return ret


class TestStaticAnalysis2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rdf_graph = get_graph('../examples/sample9.py')

    def test_example9_RANSACRegressor_succs(self):
        ret = get_query(TestStaticAnalysis2.rdf_graph, '"RANSACRegressor.linear_model.sklearn"', constructor_successors)
        expected = set(['constant:absolute_loss', 'loss:absolute_loss', 'residual_metric:[]', 'constant:0', 'constant:5',
         'max_trials:100', 'constant:50', 'constant:100', 'residual_threshold:5', 'ransac.predict', 'ransac.fit', 'logical_not.numpy',
         'min_samples:50', 'random_state:0'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example9_RANSACRegressor_preds(self):
        ret = get_query(TestStaticAnalysis2.rdf_graph, '"RANSACRegressor.linear_model.sklearn"', constructor_predecessors)
        expected = ret(['LinearRegression.linear_model.sklearn'])
        self.assertEquals(set(ret).intersection(expected), expected)


    def test_example9_LinearRegressionGD_succs(self):
        ret = get_query(TestStaticAnalysis2.rdf_graph, '"LinearRegressionGD.sample9"', constructor_successors)
        expected = set(['fit.LinearRegressionGD.sample9'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example9_LinearRegressionGD_preds(self):
        ret = get_query(TestStaticAnalysis2.rdf_graph, '"LinearRegressionGD.sample9"', constructor_predecessors)
        print(ret)
        expected = set(['StandardScaler.preprocessing.sklearn'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example9_Ridge_succs(self):
        ret = get_query(TestStaticAnalysis2.rdf_graph, '"Lasso.linear_model.sklearn"', constructor_successors)
        print(ret)
        expected = set(['constant:0.1', 'alpha:0.1', 'fit.Lasso.linear_model.sklearn', 'predict.Lasso.linear_model.sklearn'])
        self.assertEquals(set(ret).intersection(expected), expected)


class TestStaticAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rdf_graph = get_graph('../examples/sample8.py')

    def test_example8_fit(self):
        ret = get_query(rdf_graph, '"fit.PCA.decomposition.sklearn"', generic_successors)
        self.assertEqual(set(ret), set(['transform.PCA.decomposition.sklearn']))

    def test_example8_iris(self):
        ret = get_query(rdf_graph, '"load_iris.datasets.sklearn"', constructor_successors)
        print(get_graph_edges(rdf_graph))

        expected = set(['fit.PCA.decomposition.sklearn', 'transform.PCA.decomposition.sklearn', 'normalize.preprocessing.sklearn',
         'scale.preprocessing.sklearn', 'DataFrame.pandas', 'describe.df.load_iris.datasets.sklearn',
         'head.df.load_iris.datasets.sklearn', 'fit.KMeans.cluster.sklearn', 'tail.df.load_iris.datasets.sklearn',
         'fit_transform.SelectKBest.featureselection.sklearn','concat.df.load_iris.datasets.sklearn','fit.SVC.svm.sklearn'
         ])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example8_pipeline(self):
        ret = get_query(rdf_graph, '"Pipeline.pipeline.sklearn"', constructor_predecessors)
        expected = set(['PolynomialFeatures.preprocessing.sklearn', 'LinearRegression.linearmodel.sklearn'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example8_pipeline_succ(self):
        ret = get_query(rdf_graph, '"Pipeline.pipeline.sklearn"', constructor_successors)
        expected = set(['GridSearchCV.grid_search.sklearn','fit.Pipeline.pipeline.sklearn', 'predict.Pipeline.pipeline.sklearn'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example8_linearRegression(self):
        ret = get_query(rdf_graph, '"LinearRegression.linearmodel.sklearn"', constructor_successors)
        expected = set(['Pipeline.preprocessing.sklearn'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example8_polynomialfeatures(self):
        ret = get_query(rdf_graph, '"PolynomialFeatures.preprocessing.sklearn"', constructor_successors)
        expected = set(['Pipeline.preprocessing.sklearn'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example8_svc(self):
        ret = get_query(rdf_graph, '"SVC.svm"', constructor_successors)
        expected = set(['fit.SVC.svm'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example_oneclasssvm(self):
        ret = get_query(rdf_graph, '"OneClassSVM.svm"', constructor_successors)
        expected = set(['fit.OneClassSVC.svm', 'predict.OneClassSVC.svm', 'decision_function.OneClassSVC.svm'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example_oneclasssvm(self):
        ret = get_query(rdf_graph, '"KMeans.cluster.sklearn"', constructor_successors)
        expected = set(['fit.KMeans.cluster.sklearn', 'labels.KMeans.cluster.sklearn'])
        self.assertEquals(set(ret).intersection(expected), expected)

    def test_example_gridsearch(self):
        ret = get_query(rdf_graph, '"GridSearchCV.grid_search.sklearn"', constructor_successors)
        expected = set(['fit.GridSearchCV.grid_search.sklearn', '.best_params_.GridSearchCV.grid_search.sklearn'])
        self.assertEquals(set(ret).intersection(expected), expected)


