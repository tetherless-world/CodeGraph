import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
    data['results']['bindings'] = data['results']['bindings'][0:100]

    with open('../test_data/' + sys.argv[1], 'w') as f:
        json.dump(data, f)
