import sys
import re
import math
import random

def histedges_equalN(x, nbins):
    data_sorted = sorted(x, key=lambda y: y[1])
    step = math.ceil(len(data_sorted)//nbins+1)
    binned_data = []
    for i in range(0,len(data_sorted),step):
        binned_data.append(data_sorted[i:i+step])
    return binned_data

with open(sys.argv[1]) as f:
    staticData = f.readlines()
    
    matchString = '(.+) (\d+) \[(.+)\]'
    added_pairs = set()

    all_pairs = []
    all_counts = []
    all_sizes = []
    idx = 0
    for line in staticData:
        pattern = re.compile(matchString)
        adjustedLine = pattern.match(line)
        if adjustedLine == None:
            print("Found violation.")
            print(line)
        count = int(adjustedLine.group(2))
        klass = adjustedLine.group(1)
        otherClasses = adjustedLine.group(3).strip().split(', ')
        size = len(otherClasses)
        for c in otherClasses:
            p = [c, klass]
            p.sort()
            key = p[0] + '|' + p[1]
            if key in added_pairs:
                continue
            added_pairs.add(key)
            idx += 1
            all_counts.append((count, idx))
            all_sizes.append((size, idx))
            all_pairs.append((p[0], p[1], count, size))
            
    all_in_count_sample = []
    print("total counts")
    bins = histedges_equalN(all_counts, 10)
    for bin in bins:
        n = math.ceil(len(bin)/20)
        sample = random.sample(bin, n)
        print(sample)
        for i in sample:
            print(i)
            print(all_pairs[i[1]])
            all_in_count_sample.append(all_pairs[i[1]])
    print(len(all_in_count_sample))

    all_in_size_sample = []
    print("all sizes")
    bins = histedges_equalN(all_sizes, 10)
    for bin in bins:
        n = math.ceil(len(bin)/20)
        sample = random.sample(bin, n)
        print(sample)
        for i in sample:
            print(i)
            print(all_pairs[i[1]])
            all_in_size_sample.append(all_pairs[i[1]])
    print(len(all_in_size_sample))
    print(len(all_in_count_sample))
    print(len(set(all_in_size_sample).intersection(set(all_in_count_sample))))

    test_sample = set()
    test_sample.update(all_in_count_sample)
    test_sample.update(all_in_size_sample)
    
    train_sample = set(all_pairs) - set(test_sample)
    
    with open(sys.argv[2], 'w') as f:
        for i in train_sample:
            f.write(i[0] + " " + i[1] + " " + str(i[2]) + " " + str(i[3]) + "\n")

    with open(sys.argv[3], 'w') as f:
        for i in test_sample:
            f.write(i[0] + " " + i[1] + " " + str(i[2]) + " " + str(i[3]) + "\n")
