"""
  Take a set of 1.2 million files, filter those to ones that use packages we care about.  Here its sklearn
"""
import pandas as pd
import hashlib

import ast
import pandas as pd
import spacy
from nltk.tokenize import RegexpTokenizer
import FunctionCallsGatherer
import nbformat
from nbconvert import PythonExporter
import re
import sys
import os
import json

def hash_content(content):
    byte_str = str.encode(content)
    h = hashlib.md5()
    h.update(byte_str)
    digest = h.hexdigest()
    return digest

"""
    Following methods taken from: https://github.com/hamelsmu/code_search/blob/master/notebooks/1%20-%20Preprocess%20Data.ipynb
"""
def tokenize_docstring(text):
    "Apply tokenization using spacy to docstrings."
    tokens = EN.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]


def tokenize_code(text):
    "A very basic procedure for tokenizing code strings."
    return RegexpTokenizer(r'\w+').tokenize(text)


def get_function_docstring_pairs(blob):
    "Extract (function/method, docstring) pairs from a given code blob."
    pairs = []
    try:
        module = ast.parse(blob)
        classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        for _class in classes:
            functions.extend([node for node in _class.body if isinstance(node, ast.FunctionDef)])
        func2docstrings = {}

        for f in functions:
            docstring = ast.get_docstring(f) if ast.get_docstring(f) else ''
            func2docstrings[f.name] = tokenize_docstring(docstring)

        pairs.append((FunctionCallsGatherer.get_func_calls(module), func2docstrings))

    except (AssertionError, MemoryError, SyntaxError, UnicodeEncodeError):
        pass
    return pairs


def get_function_docstring_pairs_list(blob_list):
    """apply the function `get_function_docstring_pairs` on a list of blobs"""
    return [get_function_docstring_pairs(b) for b in blob_list]


def parse_as_python_nb(code):
    try:
        nb = nbformat.reads(code, as_version=4)
        exporter = PythonExporter()
        sources, resources = exporter.from_notebook_node(nb)
        source = sources
    except:
        try:
            nb = nbformat.reads(code, as_version=3)
            exporter = PythonExporter()
            sources, resources = exporter.from_notebook_node(nb)
            source = sources
        except:
            source = None
            print('failed to parse as Notebook')
            # print(code)
    # replace any awful ipython notebook style commands
    if source:
        source = re.sub(r'^%.*\n', '', source, flags=re.M)
        source = re.sub(r'^!.*\n', '', source, flags=re.M)

    return source


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
                code = df['content'].iloc[i]
                if '"cells":' in code:
                    source = parse_as_python_nb(code)
                else:
                    source = df['content'].iloc[i]
                if not source:
                    print('repo_path could not parse:' + path)
                    continue
                f.write(source)

    with open(os.path.join(sys.argv[2], "pathInfo.json"), 'w') as f:
         json.dump(path_to_path, f)
        

if __name__ == '__main__':
    main()


