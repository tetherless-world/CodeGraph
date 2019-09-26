import matplotlib.pyplot as plt
import networkx as nx
import pydotplus
import collections
import json
import re
import ast
import rdflib

Node = collections.namedtuple('Node', 'index source paths args expr data_flow_edges control_flow_edges constant_edge')

# global variable to keep track of nodes.  Note that this variable is changed by every call to add a new graph
# change if this not a behavior you want
global_node_index = 0


codegraph = rdflib.Namespace('http://purl.org/twc/codegraph/')
codegraph_activity = rdflib.Namespace('http://purl.org/twc/codegraph/activity/')
codegraph_entity = rdflib.Namespace('http://purl.org/twc/codegraph/entity/')
codegraph_tag = rdflib.Namespace('http://purl.org/twc/codegraph/tag/')
codegraph_flow_type = rdflib.Namespace('http://purl.org/twc/codegraph/flow_type/')
prov = rdflib.Namespace('http://www.w3.org/ns/prov#')


def parse_wala_into_graph(data, add_args=False):
    data_flow_edges = []
    control_flow_edges = []
    nodes = []
    constant_index = len(data)
    for index, step in enumerate(data):
        df_edges = None
        cf_edges = None

        if step['edges'] and 'DATA' in step['edges']:
            df_edges = step['edges']['DATA']
            df_edges = ast.literal_eval(df_edges)
            # first element of the list is the set of target nodes, second element is the label for each node
            for e in df_edges[0]:
                data_flow_edges.append((index, e, df_edges[1]))
        if step['edges'] and 'CONTROL' in step['edges']:
            cf_edges = step['edges']['CONTROL']
            for e in cf_edges:
                control_flow_edges.append((index, e))
        if add_args:
            if isinstance(step['args'], list):
                for idx, a in enumerate(step['args']):
                    node_type = None
                    if isinstance(a, dict):
                        # its a dictionary that got passed as an argument
                        node_type = 'dictionary'
                        constant_index += 1
                    elif isinstance(a, list) is not True:
                        node_type = 'constant'
                        constant_index += 1
                    if node_type:
                        nodes.append(Node(constant_index, step['source'], [node_type + ':' + str(a)], step['args'],
                                          step['expr'], None, None, constant_edge=(node_type, str(a))))
                        data_flow_edges.append((index, constant_index, 'constant'))
            if 'named' in step:
                named_args = step['named']
                for key in named_args:
                    constant_index += 1
                    nodes.append(Node(constant_index, step['source'], [key + ':' + str(named_args[key])], step['args'],
                                      step['expr'], None, None, constant_edge=(key, str(named_args[key]))))
                    data_flow_edges.append((index, constant_index, 'constant'))

        nodes.append(Node(index, step['source'], step['path'], step['args'], step['expr'], df_edges, cf_edges, None))

    return nodes, data_flow_edges, control_flow_edges


def show_matplot(G, nodes_in_edges, labels):
    # pos = graphviz_layout(G, prog='dot')
    pos = nx.spring_layout(G)
    if nodes_in_edges is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=list(nodes_in_edges))
    else:
        nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_cmap=plt.cm.Blues, edge_color='grey', width=2)
    nx.draw_networkx_labels(G, pos=pos, font_size=10, font_color='black', labels=labels)

    plt.show()


def gather_relevant_nodes(G, s):
    subnodes = nx.dfs_successors(G, source=s)
    ret = []
    for l in subnodes.values():
        ret.extend(l)
    for node in ret:
        prednodes = nx.dfs_predecessors(G, node)
        for l in prednodes.keys():
            ret.append(l)
    return ret


def shred_into_graphs(json_data):
    G = nx.DiGraph()
    nodes, data_flow_edges, control_flow_edges = parse_wala_into_graph(json_data)
    node_idxs = [n.index for n in nodes]
    G.add_nodes_from(node_idxs)
    G.add_edges_from(data_flow_edges)

    ug = nx.Graph()
    ug.add_nodes_from(node_idxs)
    ug.add_edges_from(data_flow_edges)

    graphs = []
    for component in nx.connected_components(ug):
        sub_graph = pydotplus.Dot(graph_type='digraph')
        sub_graph.set_node_defaults(shape="record")
        methods_used = []

        source_code = []
        for n in component:
            methods_used.append('.'.join(nodes[n].paths))
            print(nodes[n].paths)
            print(nodes[n].args)
            print(nodes[n].data_flow_edges)
            if (isinstance(nodes[n].args, list)):
                args_len = len(nodes[n].args)
                label = '<f0>R|' + nodes[n].paths[0]
                for i in range(1, args_len):
                    label += '|' + '<f' + str(i) + '>P' + str(i)
                src_node = pydotplus.Node(nodes[n].paths[0], label=label)
                sub_graph.add_node(src_node)
            line = nodes[n].expr
            matches = set(re.findall(r'\[(.*?):', line))
            int_matches = [int(s) for s in matches]
            source_code.extend(int_matches)

        # add edges now
        for s in component:
            if nodes[s].args is None:
                continue
            if isinstance(nodes[s].args, list):
                for index, a in enumerate(nodes[s].args):
                    if isinstance(a, dict):
                        # its a dictionary that got passed as an argument
                        src_node = pydotplus.Node('dictionary', label='dictionary')
                        sub_graph.add_node(src_node)
                        port = '<f' + str(index) + '>'
                        e = pydotplus.Edge(nodes[s].paths[0] + ':' + port, 'dictionary')
                        print('adding edges:' + nodes[s].paths[0] + ':' + port + '-> dictionary')
                        sub_graph.add_edge(e)
                    elif isinstance(a, list) != True:
                        # its likely just a constant that got passed as an arg
                        src_node = pydotplus.Node(str(a), label=a)
                        sub_graph.add_node(src_node)
                        port = '<f' + str(index) + '>'
                        e = pydotplus.Edge(nodes[s].paths[0] + ':' + port, str(a))
                        print('adding edges:' + nodes[s].paths[0] + ':' + port + '->' + str(a))
                        sub_graph.add_edge(e)

            for t in G.successors(s):
                if nodes[t].args is None:
                    continue

                for index, arg in enumerate(nodes[t].args):
                    if isinstance(arg, list):
                        for a in arg:
                            if '.'.join(nodes[s].paths) == '.'.join(a):
                                port = '<f' + str(index) + '>'
                                e = pydotplus.Edge(nodes[s].paths[0], nodes[t].paths[0] + ':' + port, color='green')
                                print('adding edges:' + nodes[s].paths[0] + '->' + nodes[t].paths[0] + ':' + port)
                                sub_graph.add_edge(e)

            # add any control flow edges if they exist
            if nodes[s].control_flow_edges:
                for t in nodes[s].control_flow_edges:
                    e = pydotplus.Edge(nodes[s].paths[0], nodes[t].paths[0], color='blue')
                    sub_graph.add_edge(e)

        sub_dot = sub_graph.create(format='dot')
        graphs.append((sub_dot, methods_used, sorted(source_code)))
    return graphs


def get_rdf_neighbors_as_dot(preds, succs, source):
    graph = pydotplus.Dot(graph_type='digraph')
    s = pydotplus.Node(source, label=source)

    for succ in succs:
        t = pydotplus.Node(str(succ), label=succ)
        e = pydotplus.Edge(source, t)
        graph.add_edge(e)

    for pred in preds:
        t = pydotplus.Node(str(pred), label=pred)
        e = pydotplus.Edge(t, source)
        graph.add_edge(e)

    return graph


def dump_data_as_svg(data, filename):
    graph = pydotplus.graph_from_dot_data(data)
    graph.write(filename, format='svg')


def get_svg_from_dot_data(data):
    graph = pydotplus.graph_from_dot_data(data)
    return graph.create(format='svg')


def main():
    with open('/tmp/test.json', 'r') as f:
        graphs = shred_into_graphs(json.load(f))
        for i, g in enumerate(graphs):
            dump_data_as_svg(g[0], '/tmp/test' + str(i) + '.svg')


def addToGraph(g, graph, classes_to_superclasses=None, cf_edges_to_sources={}, df_edges_to_sources={}):
    nodes = graph[0]
    # print(nodes)
    idx2node = {}
    idx2uri = {}
    global global_node_index

    for node in nodes:
        idx2node[node.index] = node
        # create a unique URI per node
        entity_uri = codegraph_entity[str(global_node_index)]
        activity_uri = codegraph_activity[str(global_node_index)]
        idx2uri[node.index] = (entity_uri, activity_uri)
        global_node_index += 1
        
        g.add((entity_uri, rdflib.RDF.type, prov.Entity))
        g.add((entity_uri, codegraph.turtle_info, rdflib.Literal(node.expr)))

        if node.constant_edge:
            g.add((entity_uri, rdflib.RDF.type, codegraph[node.constant_edge[0].title()]))
            g.add((entity_uri, prov.value, rdflib.Literal(node.constant_edge[1])))
        else:
            g.add((activity_uri, rdflib.RDF.type, prov.Activity))
            g.add((entity_uri, prov.wasGeneratedBy, activity_uri))
            
            g.add((activity_uri, codegraph.turtle_info, rdflib.Literal(node.expr)))
        
            if classes_to_superclasses:
                classes = set(classes_to_superclasses.keys())

                has_ai4_ml = set(node.paths).intersection(classes)
                if len(has_ai4_ml) > 0:
                    assert len(has_ai4_ml) == 1
                    tags = classes_to_superclasses[has_ai4_ml.pop()]
                    tags = tags[0]  # classes to superclasses has 2 levels of lists
                    for tag in tags:
                        g.add((activity_uri, rdflib.RDF.type, codegraph_tag[tag]))
            path = '.'.join(node.paths)
            g.add((activity_uri, rdflib.RDFS.label, rdflib.Literal(node.source.strip())))
            g.add((activity_uri, codegraph.path, rdflib.Literal(path)))
            g.add((activity_uri, codegraph.source, rdflib.Literal(node.source)))
            g.add((entity_uri, codegraph.source_path, rdflib.Literal(graph[3])))
        
    for edge in graph[1]: # dataflow
        if len(idx2node[edge[0]].paths) == 0 or len(idx2node[edge[1]].paths) == 0:
            continue
        src = '.'.join(idx2node[edge[0]].paths)
        target = '.'.join(idx2node[edge[1]].paths)
        # Connect the activity of the target edge to have used the entity of the source edge.
        g.add((idx2uri[edge[0]][1], prov.used, idx2uri[edge[1]][0]))
        # if len(edge) == 3:
            # this is a positional argument, add a role
            # This isn't the right parameter here, just skip for now.
            # usage = g.resource(rdflib.BNode())
            # role = g.resource(rdflib.BNode())
            # g.add((idx2uri[edge[0]][1], prov.qualifiedUsage, usage.identifier))
            # usage.add(rdflib.RDF.type, prov.Usage)
            # usage.add(prov.hadRole, role.identifier)
            # usage.add(prov.entity, idx2uri[edge[1]][0])
            # role.add(rdflib.RDF.type, codegraph.PositionalArgument)
            # role.add(prov.value, rdflib.Literal(edge[2]))
        
    for edge in graph[2]: # control flow
        if len(idx2node[edge[0]].paths) == 0 or len(idx2node[edge[1]].paths) == 0:
            continue
        src = '.'.join(idx2node[edge[0]].paths)
        target = '.'.join(idx2node[edge[1]].paths)
        # Connect the activity of the target edge to have been informed by the activity of the source edge.
        g.add((idx2uri[edge[1]][1], prov.wasInformedBy, idx2uri[edge[0]][1]))
        
    def add_edges(edges, accumulator, edge_type, invert=False):
        for edge in edges:
            if len(idx2node[edge[0]].paths) == 0 or len(idx2node[edge[1]].paths) == 0:
                continue
            src = '.'.join(idx2node[edge[0]].paths)
            target = '.'.join(idx2node[edge[1]].paths)
            if invert:
                g.add((idx2uri[edge[0]], edge_type, idx2uri[edge[1]]))
            else:
                g.add((idx2uri[edge[1]], edge_type, idx2uri[edge[0]]))
            if len(edge) == 3:
                g.add((idx2uri[edge[0]], codegraph_flow_type[str(edge[2])], idx2uri[edge[1]]))

# takes a set of control flow edges->sources, data flow edges -> sources, builds a summary of edges to counts
def create_summary_graph(cf_edges_to_sources, df_edges_to_sources):
    # create counts on edges as a graph so we can see the total number of edges and their counts
    df_edges_to_counts = {}
    for key in df_edges_to_sources:
        df_edges_to_counts[key] = len(df_edges_to_sources[key])

    cf_edges_to_counts = {}
    for key in cf_edges_to_sources:
        cf_edges_to_counts[key] = len(cf_edges_to_sources[key])

    edges = []
    for key in df_edges_to_counts:
        if 'constant' in key:
            continue
        if 'dictionary' in key:
            continue
        e = {}
        e["source"] = key[0]
        e["target"] = key[2]
        e["flow_type"] = "data_flow"
        e["flows_to"] = key[1]
        e["count"] = df_edges_to_counts[key]
        edges.append(e)
    for key in cf_edges_to_counts:
        e = {}
        e["source"] = key[0]
        e["target"] = key[1]
        e["flow_type"] = "control_flow"
        e["count"] = cf_edges_to_counts[key]
        edges.append(e)
    return edges


if __name__ == '__main__':
    main()
