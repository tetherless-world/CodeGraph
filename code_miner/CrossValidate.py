import sys
import os
import CollectAllGraphs
import StaticAnalysisGraphBuilder
import rdflib
import os

url = 'http://localhost:4567/analyze_code'


def build_train_edges(test, train_dir, out_dir, do_debug = False, train_file_id = 0, debug = 0, debug_limit = None):
    df_edges_to_sources = {}
    cf_edges_to_sources = {}

    # build a hash map over all edges in the training graphs except for the test files
    train_files = []
    g = rdflib.Graph()

    for f in os.listdir(train_dir):
        if not f.startswith('sample'):  
            continue
        if f in test:
            print("Skipping:" + f)
            continue
        if do_debug and debug_limit and debug >= debug_limit:
            continue
        train_files.append(f)
        debug += 1
        with open(os.path.join(train_dir, f)) as sample_file:
            source = sample_file.read()
            graph_tuple = CollectAllGraphs.handle_call_to_analysis(source, f)
            if graph_tuple:
                classes_to_superclasses = None
                StaticAnalysisGraphBuilder.addToGraph(g, graph_tuple, classes_to_superclasses, cf_edges_to_sources,
                                                  df_edges_to_sources)
    train_files.sort()

    with open(out_dir + str(train_file_id), 'w') as o:
        for train_file in train_files:
            o.write(train_file + '\n')

    df = list(df_edges_to_sources.keys())

    with open(os.path.join(out_dir, 'train_data_edges' + str(train_file_id)), 'w') as o:
        for edge in df:
            o.write(str(edge) + '\n')
    g.serialize(destination=os.path.join(out_dir, 'train_graph.ttl'), format='turtle')
    return cf_edges_to_sources, df_edges_to_sources



def check_test(test, cf_edges_to_sources, df_edges_to_sources, do_debug = False, debug_limit = None):
    # now walk over each test file and see if the edges in the test file are present in the train summary
    num_cf_passes = 0
    num_cf_fails = 0
    num_df_passes = 0
    num_df_fails = 0
    debug = 0
    for f in test:
        if do_debug and debug_limit and debug >= debug_limit:
            continue
        debug += 1
        with open(os.path.join(sys.argv[2], f)) as sample_file:
            source = sample_file.readlines()
            source = '\n'.join(source)
            graph_tuple = CollectAllGraphs.handle_call_to_analysis(source, f)
            if graph_tuple:
                classes_to_superclasses = None
                g = rdflib.Graph()
                test_df_edges = {}
                test_cf_edges = {}
                StaticAnalysisGraphBuilder.addToGraph(g, graph_tuple, classes_to_superclasses, test_cf_edges,
                                                      test_df_edges)
                o = os.path.splitext(os.path.basename(f))[0]
                out = os.path.join(sys.argv[3], 'test_graph_' + o + '.ttl')
                print(out)
                g.serialize(out, 'turtle')

                for key in test_cf_edges:
                    if key in cf_edges_to_sources:
                        num_cf_passes += 1
                    else:
                        num_cf_fails += 1

                for key in test_df_edges:
                    if key in df_edges_to_sources:
                        num_df_passes += 1
                    else:
                        num_df_fails += 1

                if do_debug:
                    print("******** DATA FLOW**********")
                    print("TRAIN")
                    for key in df_edges_to_sources:
                        print(key)

                    print("TEST")
                    for key in test_df_edges:
                        print(key)

                    print("******** CONTROL FLOW**********")
                    print("TRAIN")
                    for key in cf_edges_to_sources:
                        print(key)

                    print("TEST")
                    for key in test_cf_edges:
                        print(key)

    print("Number of cf passes:" + str(num_cf_passes))
    print("Number of cf fails:" + str(num_cf_fails))
    print("Number of df passes:" + str(num_df_passes))
    print("Number of df fails:" + str(num_df_fails))


def dump_all_training(test_dir):
    test_files = []
    for f in os.listdir(test_dir):
        if f.startswith('test'):
            test_files.append(f)

    train_dir = os.path.dirname(os.path.dirname(test_dir))
    i = 0
    prev_test = []
    for f in test_files:
        with open(os.path.join(test_dir, f)) as tf:
            test = tf.readlines()
            test = [s.rstrip() for s in test]
            test.sort()
            assert prev_test != test
            prev_test = test
            build_train_edges(test, train_dir, True, i)
            i += 1


def main():
    if len(sys.argv) == 2:
        dump_all_training(sys.argv[1])
    else:
        with open(sys.argv[1]) as f:
            test = f.readlines()
        test = [s.rstrip() for s in test]
        print("training")
        cf_edges_to_sources, df_edges_to_sources = build_train_edges(test, sys.argv[2], sys.argv[3])
        print("testing")
        check_test(test, cf_edges_to_sources, df_edges_to_sources)


if __name__ == '__main__':
    main()