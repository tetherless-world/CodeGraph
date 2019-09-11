import os
import sys
import Python_File_Miner

i = 0
for f in os.listdir(sys.argv[1]):
    print(f)
    with open(os.path.join(sys.argv[1], f)) as source_file:
        source = source_file.read()
        if f.endswith('.ipynb'):
            source = Python_File_Miner.parse_as_python_nb(source)
        if not source:
            continue
        with open(sys.argv[2] + str(i) + '.py', 'w') as of:
            of.write(source)
        i += 1

