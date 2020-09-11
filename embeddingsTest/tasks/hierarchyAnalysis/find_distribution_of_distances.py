import pickle
import sys

with open(sys.argv[1], 'rb') as f:
    len_to_paths = pickle.load(f)
    print('Distance' + '\t' + 'Count')
    for i in len_to_paths:
        print(str(i) + '\t' + str(len(len_to_paths[i])))
