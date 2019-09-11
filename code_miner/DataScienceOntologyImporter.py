import rdflib


class DataScienceOntologyImporter(object):
    def __init__(self):

        graph = rdflib.Graph()
        graph.parse("../data/dso.ttl", format='turtle')

        self.classes2KeyText = {}
        self.classesToSuperclasses = {}

        with open("../dso_queries.sparql", 'r') as f:
            query_str = f.read()
            qres = graph.query(query_str)

            for row in qres:
                clazz = str(row[0])
                if clazz in self.classes2KeyText:
                    text = self.classes2KeyText[clazz]
                else:
                    text = set([])
                    self.classes2KeyText[clazz] = text
                text.add(str(row[1]))
                text.add(str(row[2]))
                text.add(str(row[4]))
                text.add(str(row[5]))

                if clazz in self.classesToSuperclasses:
                    superclasses = self.classesToSuperclasses[clazz]
                else:
                    superclasses = []
                    self.classesToSuperclasses[clazz] = superclasses

                superclasses.append(str(row[3]))

    def get_classes_2_superclasses(self):
        return self.classesToSuperclasses

    def get_classes_2_key_text(self):
        return self.classes2KeyText

def main():
    dso = DataScienceOntologyImporter()
    print(dso.get_classes_2_key_text())
    print(dso.get_classes_2_superclasses())

if __name__ == '__main__':
    main()

