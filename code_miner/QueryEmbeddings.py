import faiss
import numpy
import sys

def compute_all_class_NN(embeddings, index, methods):
    # I contains a 2D array - rows for each query, columns being the k
    # D contains a 2D array of distances
    D, I = index.search(embeddings, 10)
    print(I.shape)
    print(D.shape)
    for row_index, row in enumerate(I):
        print(row)
        print("class:" + methods[row_index])
        for nn_num, nn_index in enumerate(row):
            if nn_index == -1:
                continue
            print(D[row_index][nn_num])
            print(methods[nn_index])


def main():
    # precreated faiss index
    index = faiss.read_index(sys.argv[1])
    # embeddings to use for querying
    embeddings = numpy.load(sys.argv[2])
    # text that corresponds to each embeddings in arg[2]
    with open(sys.argv[3], 'r') as out:
        methods = out.readlines()

    compute_all_class_NN(embeddings, index, methods)


if __name__ == '__main__':
    main()

