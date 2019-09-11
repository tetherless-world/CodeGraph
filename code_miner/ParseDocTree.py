import docutils.nodes
import docutils.parsers.rst
import docutils.utils
import re
import os
import json

class Visitor(docutils.nodes.NodeVisitor):

    def __init__(self, doc):
        super().__init__(doc)
        self.objs = []
        self.current_obj = None

    def visit_reference(self, node: docutils.nodes.reference) -> None:
        """Called for "reference" nodes."""
        pass

    def unknown_visit(self, node: docutils.nodes.Node) -> None:
        """Called for all other node types."""
        """ need: target (target.names), section ->children->paragraph  """
        if node.tagname == 'title':
            if self.current_obj is not None:
                self.objs.append(self.current_obj)
            self.current_obj = {}
            self.current_obj['title'] = node.rawsource
            if 'names' in node.parent.attributes:
                self.current_obj['name'] = node.parent.attributes['names']
        if node.tagname == 'paragraph':
            if self.current_obj is not None:
                if 'description' in self.current_obj:
                    self.current_obj['description'] = self.current_obj['description'] + '\n' + node.rawsource
                else:
                    self.current_obj['description'] = node.rawsource


def parse_rst(text: str) -> docutils.nodes.document:
    parser = docutils.parsers.rst.Parser()
    components = (docutils.parsers.rst.Parser,)
    settings = docutils.frontend.OptionParser(components=components).get_default_values()
    document = docutils.utils.new_document('<rst-doc>', settings=settings)
    parser.parse(text, document)
    return document


def process_classes(objs, all_results):
    for obj in objs:
        if 'description' not in obj:
            continue
        class_ref = re.findall(r':class:`([^`]*)', obj['description'])
        class_ref = set(class_ref)
        # we may have multiple class references here. In this case, just accumulate results
        if class_ref is not None:
            for c in class_ref:
                if '.' in c:
                    clazz = c.split('.')[-1]
                if c in all_results:
                    all_results[c].append(obj)
                else:
                    all_results[c] = [obj]

def process_file(root, f, all_results):
    with open(os.path.join(root, f), 'r', errors='ignore') as file:
        content = file.read()
        doc = parse_rst(content)
        visitor = Visitor(doc)
        doc.walk(visitor)
        process_classes(visitor.objs, all_results)


def get_class_desc_from_rst():
    sk = "/Users/kavithasrinivas/code/scikit-learn/doc/modules"
    print(os.listdir(sk))
    all_results = {}
    for root, subdirs, files in os.walk(sk):
        for f in files:
            process_file(root, f, all_results)
    return all_results


if __name__ == "__main__":
    # print(json.dumps(get_class_desc_from_rst(), indent=4))
    all_results = {}
    process_file('/Users/kavithasrinivas/code/scikit-learn/doc/modules', 'decomposition.rst',
                                  all_results)
    print(json.dumps(all_results, indent=4))
