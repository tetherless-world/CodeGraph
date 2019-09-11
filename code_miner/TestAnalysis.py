import requests
import json
import VisualizeStaticAnalysis

input_file = '../examples/sample8.py'
url = 'http://localhost:4567/analyze_code_all'
data = open(input_file, 'rb').read()
res = requests.post(url=url,
                    data=data,
                    headers={'Content-Type': 'application/octet-stream'})
print(res.text)
json_data = json.loads(res.text)
print(json.dumps(json_data, indent=4))
graphs = VisualizeStaticAnalysis.shred_into_graphs(json_data)
with open('../examples/sample8.py', 'r') as f:
    lines = f.readlines()

for i, g in enumerate(graphs):
    VisualizeStaticAnalysis.dump_data_as_svg(g[0], '/tmp/sample' + str(i) + '.svg')
    # corresponding source code
    source_code = g[2]
    with open('/tmp/sample' + str(i) + '.py', 'w') as out:
        for line in source_code:
            print(lines[line-1])
            out.write(lines[line - 1] + '\n')


