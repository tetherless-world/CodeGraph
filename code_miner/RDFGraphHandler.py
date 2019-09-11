import rdflib
import rdflib.plugins.sparql as sparql
# from code_miner import VisualizeStaticAnalysis

class RDFGraphHandler(object):

    def __init__(self, graph=None):
        with open("../static_analysis_queries/generic_successors.sparql", 'r') as f:
            query_str = f.read()
        self.successors_query = sparql.prepareQuery(query_str)

        with open("../static_analysis_queries/constructor_successors.sparql", 'r') as f:
            query_str = f.read()
        self.constructor_successors_query = sparql.prepareQuery(query_str)

        with open("../static_analysis_queries/generic_predecessors.sparql", 'r') as f:
            query_str = f.read()
        self.predecessors_query = sparql.prepareQuery(query_str)

        if not graph:
            self.graph = rdflib.Graph()
            self.graph.parse("../data/rdfGraph.ttl", format='turtle')
        else:
            self.graph = graph

    def query(self, method_name, successors=True, constructors=False):
        if successors and not constructors:
            q = self.successors_query
        elif successors and constructors:
            q = self.constructor_successors_query
        else:
            q = self.predecessors_query
        method = rdflib.Literal(method_name)
        results = self.graph.query(q, initBindings={'m': method})
        ret = []
        for row in results:
            ret.append((row[0].value, row[1].value))
        return ret

    def get_graph_for_method(self, method_name):
        source = method_name
        succs = self.query(source, successors=True)
        preds = self.query(source, successors=False)
        sorted(succs)
        sorted(preds)
        succs = succs[-5:]
        preds = preds[-5:]
        graph = VisualizeStaticAnalysis.get_rdf_neighbors_as_dot(preds, succs, source)
        return graph

def main():
    graph_handler = RDFGraphHandler()
    graph = graph_handler.get_graph_for_method()
    graph.write('/tmp/testrdf', format='svg')


if __name__ == '__main__':
    main()


