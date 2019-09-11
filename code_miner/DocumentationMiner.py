import hashlib
import pandas as pd
import sys
import os
import json

def hash_content(content):
    byte_str = str.encode(content)
    h = hashlib.md5()
    h.update(byte_str)
    digest = h.hexdigest()
    return digest


def main():
    sample_number = 0
    path_to_path = {}
    hash_values = set([])

    for csv_file in os.listdir(sys.argv[1]):
        df = pd.read_csv(os.path.join(sys.argv[1], csv_file), compression='gzip', header=0)
        # print(df)
        for i in range(0, len(df)):
            hash_v = df.content.apply(hash_content)
            if hash_v in hash_values:
                continue
            hash_values.append(hash_v)
            path = df['repo_path'].iloc[i]
            file_name = 'sample' + sample_number + '.py'
            sample_number += 1
            path_to_path[file_name] = path

            with open(os.path.join(sys.argv[2], file_name), 'w') as f:
                source = df['content'].iloc[i]
                f.write(source)

    with open(os.path.join(sys.argv[2], "pathInfo.json"), 'w') as f:
        json.dump(path_to_path, f)


if __name__ == '__main__':
    main()