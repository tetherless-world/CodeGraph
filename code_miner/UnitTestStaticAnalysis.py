import unittest
import requests
import json
import rdflib
import StaticAnalysisGraphBuilder

url = 'http://localhost:4567/analyze_code'

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
    nodes, data_flow_edges, control_flow_edges = StaticAnalysisGraphBuilder.parse_wala_into_graph(json_data, add_args=False)
    g = (nodes, data_flow_edges, control_flow_edges, f)
    rdf_graph = rdflib.Graph()
    StaticAnalysisGraphBuilder.addToGraph(rdf_graph, g)
    return rdf_graph

class UnitTestStaticAnalysis(unittest.TestCase):
    """
       Cant reuse methods in RDFHandler because they use prepared queries which do not seem to work against
       a dynamically created graph.  Most annoying
    """
    def check_edge(self, rdf_graph, query, source, target):
        with open(query, 'r') as f:
            query_str = f.read()
        query_str = query_str.replace('?src', source)
        query_str = query_str.replace('?target', target)
        results = rdf_graph.query(query_str)
        print(results)
        ret = []
        for row in results:
            self.assertTrue(row)

    def test_example1(self):
        rdf_graph = get_graph('../examples/unittest1.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                                             '"load_iris.datasets.sklearn"', '"train_test_split.cross_validation.sklearn"')

    def test_example2(self):
        rdf_graph = get_graph('../examples/unittest1.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"SVC.svm.sklearn"', '"fit.SVC.svm.sklearn"')

    def test_example3(self):
        rdf_graph = get_graph('../examples/unittest1.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                         '"train_test_split.cross_validation.sklearn"', '"fit.SVC.svm.sklearn"')

    def test_example4(self):
        rdf_graph = get_graph('../examples/unittest2.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"load_iris.datasets.sklearn"', '"DataFrame.pandas"')

    def test_example5(self):
        rdf_graph = get_graph('../examples/unittest2.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"Dataframe.pandas"', '"head.DataFrame.pandas"')

    def test_example6(self):
        rdf_graph = get_graph('../examples/unittest2.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"DataFrame.pandas"', '"describe.DataFrame.pandas"')

    def test_example7(self):
        rdf_graph = get_graph('../examples/unittest3.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                                             '"load_iris.datasets.sklearn"', '"scale.preprocessing.sklearn"')

    def test_example8(self):
        rdf_graph = get_graph('../examples/unittest3.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                                    '"scale.preprocessing.sklearn"', '"DataFrame.pandas"')

    def test_example9(self):
        rdf_graph = get_graph('../examples/unittest3.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                                    '"DataFrame.pandas"', '"head.DataFrame.pandas"')

    def test_example10(self):
        rdf_graph = get_graph('../examples/unittest4.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                                    '"load_iris.datasets.sklearn"', '"fit.PCA.decomposition.sklearn"')

    def test_example11(self):
        rdf_graph = get_graph('../examples/unittest4.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                                    '"PCA.decomposition.sklearn"', '"fit.PCA.decomposition.sklearn"')

    def test_example12(self):
        rdf_graph = get_graph('../examples/unittest5.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                                    '"load_iris.datasets.sklearn"', '"fit_transform.SelectKBest.feature_selection.sklearn"')

    def test_example13(self):
        rdf_graph = get_graph('../examples/unittest5.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                                    '"SelectKBest.feature_selection.sklearn"', '"fit_transform.SelectKBest.feature_selection.sklearn"')


    def test_example13(self):
        rdf_graph = get_graph('../examples/unittest6.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                    '"DataFrame.pandas"', '"fit_transform.LabelEncoder.preprocessing.sklearn"')


    def test_example14(self):
        rdf_graph = get_graph('../examples/unittest6.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"fit_transform.LabelEncoder.preprocessing.sklearn"', '"inverse_transform.LabelEncoder.preprocessing.sklearn"' )


    def test_example15(self):
        rdf_graph = get_graph('../examples/unittest6.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"inverse_transform.LabelEncoder.preprocessing.sklearn"', '"DataFrame.pandas"')


    def test_example16(self):
        rdf_graph = get_graph('../examples/unittest6.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"toarray.fit_transform.OneHotEncoder.preprocessing.sklearn"', '"DataFrame.pandas"')


    def test_example17(self):
        rdf_graph = get_graph('../examples/unittest7.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"load_iris.datasets.sklearn"', '"DataFrame.pandas"')


    def test_example18(self):
        rdf_graph = get_graph('../examples/unittest7.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"randint.random.numpy"', '"DataFrame.pandas"')


    def test_example19(self):
        rdf_graph = get_graph('../examples/unittest7.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"DataFrame.pandas"', '"concat.pandas"')

    def test_example20(self):
        rdf_graph = get_graph('../examples/unittest7.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"concat.pandas"', '"head.concat.pandas"')

    def test_example21(self):
        rdf_graph = get_graph('../examples/unittest8.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"linspace.numpy"', '"meshgrid.numpy"')

    def test_example22(self):
        rdf_graph = get_graph('../examples/unittest8.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"randn.random.numpy"', '"r_.numpy"')


    def test_example23(self):
        rdf_graph = get_graph('../examples/unittest8.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                   '"r_.numpy"', '"fit.OneClassSVM.svm.sklearn"')


    def test_example24(self):
        rdf_graph = get_graph('../examples/unittest8.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"r_.numpy"', '"predict.OneClassSVM.svm.sklearn"')


    def test_example25(self):
        rdf_graph = get_graph('../examples/unittest8.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"meshgrid.numpy"', '"ravel.meshgrid.numpy"')

    def test_example26(self):
        rdf_graph = get_graph('../examples/unittest8.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"ravel"', '"c_.numpy"')

    def test_example27(self):
        rdf_graph = get_graph('../examples/unittest8.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"c_.numpy"', '"decision_function.OneClassSVM.svm.sklearn"')

    def test_example28(self):
        rdf_graph = get_graph('../examples/unittest9.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"load_iris.datasets.sklearn"', '"fit.KMeans.cluster.sklearn"')


    def test_example29(self):
        rdf_graph = get_graph('../examples/unittest9.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"KMeans.cluster.sklearn"', '"labels_.KMeans.cluster.sklearn"')

    def test_example30(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"rand.random.numpy"', '"sort.numpy"')

    def test_example31(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"pi.numpy"', '"cos.numpy"')

    def test_example32(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"cos.numpy"', '"array.numpy"')

    def test_example33(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"PolynomialFeatures.preprocessing.sklearn"', '"Pipeline.pipeline.sklearn"')

    def test_example34(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"LinearRegression.linearmodel.sklearn"', '"Pipeline.pipeline.sklearn"')

    def test_example35(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"LinearRegression.linearmodel.sklearn"', '"Pipeline.pipeline.sklearn"')

    def test_example35(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"newaxis.numpy"', '"sort.numpy"')

    def test_example36(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"sort.numpy"', '"fit.Pipeline.pipeline.sklearn"')

    def test_example37(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"array.numpy"', '"fit.Pipeline.pipeline.sklearn"')

    def test_example38(self):
        rdf_graph = get_graph('../examples/unittest10.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"array.numpy"', '"fit.Pipeline.pipeline.sklearn"')

    def test_example39(self):
        rdf_graph = get_graph('../examples/unittest11.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"dict"', '"GridSearchCV.grid_search.sklearn"')

    def test_example40(self):
        rdf_graph = get_graph('../examples/unittest11.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"Pipeline.pipeline.sklearn"', '"GridSearchCV.grid_search.sklearn"')

    def test_example41(self):
        rdf_graph = get_graph('../examples/unittest12.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"read_csv.pandas"', '"StandardScaler.preprocessing.sklearn"')

    def test_example42(self):
        rdf_graph = get_graph('../examples/unittest12.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"LinearRegressionGD"', '"fit.LinearRegressionGD"')

    def test_example43(self):
        rdf_graph = get_graph('../examples/unittest13.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"read_csv.pandas"', '"fit.RANSACRegressor.linear_model.sklearn"')

    def test_example44(self):
        rdf_graph = get_graph('../examples/unittest14.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"asarray.numpy"', '"fit.ElasticNet.linear_model.sklearn"')

    def test_example45(self):
        rdf_graph = get_graph('../examples/unittest14.py')
        self.check_edge(rdf_graph, "../static_analysis_queries/exists_edge.sparql",
                        '"asarray.numpy"', '"mean_squared_error.metrics.sklearn"')

