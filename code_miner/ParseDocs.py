import inspect
import sklearn as sk
import re
import pkgutil
import sys
import ParseDocTree
import AI4MLTagReader
import argparse
from inspect import signature
import DataScienceOntologyImporter
import MongoDBHandler as mongo


classes_to_rst = ParseDocTree.get_class_desc_from_rst()
classes_to_superclasses = AI4MLTagReader.get_class_hierarchy()
class_init_to_parameters = AI4MLTagReader.get_class_parameter_expressions()
classes_init_to_parameter_ranges = AI4MLTagReader.get_class_parameter_ranges()
classes_init_to_parameter_ranges = AI4MLTagReader.get_class_parameter_ranges()
dso = DataScienceOntologyImporter.DataScienceOntologyImporter()
classes_to_dso_superclasses = dso.get_classes_2_superclasses()
classes_to_dso_key_text = dso.get_classes_2_key_text()
mongo_handler = mongo.MongoDBHandler()


class Method_Description:

    def __init__(self, clazz, class_doc, method, method_doc, param_map, return_map, overall_doc):
        self.clazz = clazz
        self.class_doc = class_doc
        self.method = method
        self.method_doc = method_doc
        self.param_map = param_map
        self.return_map = return_map
        self.overall_doc = overall_doc
        global classes_to_rst
        global classes_to_superclasses
        global class_init_to_parameters

        class_name = re.sub(r"[<>']", '', str(clazz))
        class_name = class_name.replace('class', '')
        class_name = class_name.strip()

        self.class_extra_doc = None
        if clazz.__name__ in classes_to_rst:
            self.class_extra_doc = classes_to_rst[clazz.__name__]
        self.ai4mltags = None
        if class_name in classes_to_superclasses:
            l = classes_to_superclasses[class_name][0]['py/tuple']
            self.ai4mltags = l
        self.init_params_doc = None
        if method == '__init__' and clazz.__name__ in class_init_to_parameters:
            self.init_params_doc = class_init_to_parameters[clazz.__name__]
        self.init_param_ranges = None
        if method == '__init__' and clazz.__name__ in classes_init_to_parameter_ranges:
            self.init_param_ranges = classes_init_to_parameter_ranges[clazz.__name__]
        self.dso_superclasses = None


        if class_name in classes_to_dso_superclasses:
            self.dso_superclasses = classes_to_dso_superclasses[class_name]
        self.dso_keytext = None
        if class_name in classes_to_dso_superclasses:
            self.dso_keytext = list(classes_to_dso_key_text[class_name])

    def __str__(self):
        return str(self.clazz)+'\n' + self.class_doc + '\n' + \
               'METHOD:' + str(self.method) + '\n' + \
               self.method_doc + '\n' + str(self.param_map) + '\n' + str(self.return_map) + '\n' + self.overall_doc \
               + '\n' + self.class_extra_doc + '\n' + str(self.ai4mltags) + '\n' + str(self.init_params_doc + '\n'
               + str(self.init_param_ranges) + '\n' + self.dso_superclasses + '\n' + self.dso_keytext)

    def to_dict(self):
        ret = {}
        ret['class'] = str(self.clazz)
        ret['class_name'] = self.clazz.__name__
        ret['class_doc'] = self.class_doc
        ret['class_usage_doc'] = self.class_extra_doc
        ret['method'] = str(self.method)
        ret['method_doc'] = self.method_doc
        ret['param_map'] = self.param_map
        ret['return_map'] = self.return_map
        ret['overall_doc'] = self.overall_doc
        ret['ai4mltags'] = self.ai4mltags
        ret['init_params'] = self.init_params_doc
        ret['init_param_ranges'] = self.init_param_ranges
        ret['dso_super_classes'] = self.dso_superclasses
        ret['dso_keytext'] = self.dso_keytext
        return ret


def inspect_all(f):
    if inspect.ismodule(f):
        return inspect_module(f)
    elif inspect.isclass(f):
        return inspect_class(f)


def inspect_module(f):
    module_to_classes = {}
    module_to_classes[f] = []
    for c_name, c in inspect.getmembers(f, inspect.isclass):
        module_to_classes[f].append(inspect_class(c))
    return module_to_classes


def inspect_class(f):
    class_to_methods = {}
    class_to_methods[f] = []
    for m_name, m in inspect.getmembers(f, inspect.isfunction):
        class_to_methods[f].append((m, m_name, inspect.getfullargspec(m), inspect.getdoc(m)))
    for m_name, m in inspect.getmembers(f, inspect.ismethod):
        class_to_methods[f].append((m, m_name, inspect.getfullargspec(m), inspect.getdoc(m)))
    return class_to_methods


def inspect_module_sub_package(module):
    method_descriptions = []
    modules = inspect_all(module)
    m = inspect.getmembers(module, inspect.isfunction)

    for function in m:
        overall_doc = inspect.getdoc(function[1])
        sig = signature(function[1])
        param_names = list(sig.parameters)
        method_doc = None
        param_doc = None
        param_map = None
        return_map = None

        if overall_doc is not None:
            method_doc, param_doc, returns_doc = getDocStructure(overall_doc)
            param_map = create_parameter_map(param_doc, param_names)
            return_map = create_returns_map(returns_doc)

        method_descriptions.append(
            Method_Description(module, None, function[1].__name__, method_doc, param_map, return_map,
                               overall_doc).to_dict())

    for _, classes in modules.items():
        for c in classes:
            for clazz, methods in c.items():
                class_doc = clazz.__doc__
                if class_doc is not None and type(class_doc) is str:
                    class_doc, param_doc, returns_doc = getDocStructure(class_doc)

                for method in methods:
                    m = method[1]

                    # gather all parameter names
                    param_names = []
                    param_names.extend(method[2].args)
                    if method[2].varargs is not None:
                        param_names.extend(method[2].varargs)
                    if method[2].varkw is not None:
                        param_names.extend(method[2].args)

                    if 'self' in param_names:
                        param_names.remove('self')

                    if method == '__init__':
                        param_map = None
                        if param_doc is not None:
                            param_map = create_parameter_map(param_doc, param_names)

                        method_descriptions.append(
                            Method_Description(clazz, class_doc, m, None, param_map, None, class_doc).to_dict())
                        continue

                    if method[3] is None:
                        continue

                    overall_doc = method[3]

                    method_doc, param_doc, returns_doc = getDocStructure(overall_doc)
                    param_map = create_parameter_map(param_doc, param_names)
                    return_map = create_returns_map(returns_doc)

                    method_descriptions.append(
                        Method_Description(clazz, class_doc, m, method_doc, param_map, return_map,
                                           overall_doc).to_dict())
    return method_descriptions


def create_returns_map(returns_doc):
    return_map = {}
    returns_list = []

    if returns_doc is not None:
        sentences = returns_doc.split('\n')
        for s in sentences:
            if s.find(':') != -1:
                potential_ret = re.findall(r'(.*):', s )
                if len(potential_ret[0].split()) > 1:
                    continue
                returns_list.append(potential_ret[0])

        for r in returns_list:
            r = r.strip()
            # mongodb does not like keys with '.' in their names
            if '.' in r:
                r = r.replace('.', '/')
            sentences = returns_doc.split('\n')
            for s in sentences:
                s.strip()

                if s.startswith(r):
                    ret_obj = {}
                    return_map[r] = ret_obj
                    val = re.findall(':(.*)', s, re.DOTALL)
                    if val is None or len(val) == 0:
                        continue
                    val = val[0]

                    t = find_type(val)
                    if t is not None:
                        ret_obj['type'] = t

                    if 'shape' in val:
                        shape = find_shape(val, False)
                        ret_obj['dimensionality'] = shape

                    ret_obj['doc'] = val

                else:
                    if r in return_map:
                        ret_obj = return_map[r]

                        if 'doc' in ret_obj:
                            ret_obj['doc'] = ret_obj['doc'] + '\n' + s
                        else:
                            ret_obj['doc'] = s

    return return_map

def find_type(param_str):
    t = re.findall(r'int|float|map|dict|str|bool|array|list|set|tuple', param_str)
    if t is not None and len(t) > 0:
        return t[0]
    else:
        t = param_str.strip().split()
        if len(t) > 0:
            return t[0]
    return None


def find_optional(param_str):
    return param_str.find('optional') > -1


def find_shape(param_str, first=True):
    if first:
        pattern = r'shape\s*=?\s*[\(\[{](.*)[\)\]}]'
    else:
        pattern = r'[\(\[{](.*)[\)\]}]'
    shapes = re.findall(pattern, param_str)
    dims = 0
    if shapes is not None and len(shapes) > 0:
        dimensions = shapes[0].split(',')
        dims = len(dimensions)
        if len(dimensions) > 1 and dimensions[len(dimensions) - 1] == '':
            dims -= 1
    return dims


def create_parameter_map(param_doc, param_names):
    param_map = {}
    param_start = {}

    if param_doc is not None:
        for p in param_names:
            p_start = re.search(p + '\s?:', param_doc)
            if p_start:
                param_start[p] = p_start.start()
    sorted_l = sorted(param_start.values())
    for p in param_start:
        assert param_start[p] in sorted_l
        index = sorted_l.index(param_start[p])
        if index < len(sorted_l) - 1:
            param_end = sorted_l[sorted_l.index(param_start[p]) + 1]
        else:
            param_end = len(param_doc)
        param_str = param_doc[param_start[p] + len(p + ' :'): param_end]
        param_obj = {}

        param_obj['name'] = p
        param_obj['param_doc'] = param_str
        t = find_type(param_str)
        if t is not None:
            param_obj['type'] = t

        t = find_optional(param_str)
        if t:
            param_obj['optional'] = t

        if 'shape' in param_str:
            shapes = []
            prev = 0
            for m in re.finditer('[\)\]}]', param_str):
                shapes.append(param_str[prev:m.end()])
                prev = m.end() + 1

            if len(shapes) == 1:
                param_obj["dimensionality"] = [find_shape(param_str)]
            elif len(shapes) > 1:
                dims = []
                for i, m in enumerate(shapes):
                    first = True
                    if i > 0:
                        first = False
                    dim = find_shape(m, first)
                    if dim > 0:
                        dims.append(dim)
                param_obj["dimensionality"] = list(set(dims))

        param_map[str(param_names.index(p))] = param_obj
    return param_map


def getDocStructure(overall_doc):
    #weird stuff here
    if type(overall_doc) is not str:
        return
    param_finds = [m.start() for m in re.finditer('Parameters', overall_doc)]
    return_finds = [m.start() for m in re.finditer('Returns', overall_doc)]

    # if we have multiple parameters and multiple returns, we are likely in a very odd situation
    # another possibility is that returns is used in a different context before parameters
    if len(param_finds) == 1 and len(return_finds) == 1 and param_finds[0] < return_finds[0]:
        if overall_doc.find('Parameters') != -1:
            method_doc = re.findall('(.*)Parameters', overall_doc, re.DOTALL)[0]
            if overall_doc.find('Returns') != -1:
                param_doc = re.findall('Parameters(.*)Returns', overall_doc, re.DOTALL)[0]
                returns_doc = re.findall('Returns(.*)', overall_doc, re.DOTALL)[0]
                see_also_finds = returns_doc.lower().rfind('see also')
                if see_also_finds != -1:
                    returns_doc = returns_doc[0:see_also_finds]
            else:
                param_doc = re.findall('Parameters(.*)', overall_doc, re.DOTALL)[0]
                returns_doc = None
        else:
            method_doc = overall_doc
            param_doc = None
            returns_doc = None

    else:
        param_doc = None
        returns_doc = None

        param_start = overall_doc.rfind('Parameters')
        return_start = overall_doc.rfind('Returns')
        if param_start != -1:
            method_doc = overall_doc[0:param_start]

            if param_start > return_start:
                param_doc = overall_doc[param_start + len('Parameters'):]
                return method_doc, param_doc, returns_doc

            if return_start != -1:
                param_doc = overall_doc[param_start + len('Parameters'): return_start]
                see_also_finds = overall_doc.lower().rfind('see also')

                if see_also_finds != -1:
                    returns_doc = overall_doc[return_start + len('Returns'):see_also_finds]
                else:
                    returns_doc = overall_doc[return_start + len('Returns'):]
            else:
                param_doc = overall_doc[param_start:]
        else:
            method_doc = overall_doc
    return method_doc, param_doc, returns_doc


def main():
    package = sk
    collection = 'sklearn'
    mongo_handler.drop_db(collection)

    for importer, modname, ispkg in pkgutil.walk_packages(path=package.__path__,
                                                          prefix=package.__name__+'.',
                                                          onerror=lambda x: print(x)):
        try:
            print(modname)
            module = importer.find_module(modname).load_module(modname)
            ret_json = inspect_module_sub_package(module)
            if len(ret_json) > 0:
                mongo_handler.insert_many(collection, ret_json)
        except ModuleNotFoundError:
            print('could not load module', sys.exc_info()[0])
        except TypeError:
            print('type hierarchy of module is not correct', sys.exc_info()[0])

    mongo_handler.create_text_index(collection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the documentation for sklearn and write it into a mongodb - this code assumes mongo db has been started with an empty dir")
    parsed_args = parser.parse_args()
    main()


