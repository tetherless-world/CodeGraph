import json


def load_json(f):

    with open(f) as f:
        return json.load(f)


def get_class_hierarchy():
    return load_json('classes_to_tags.json')


def get_class_parameter_expressions():
    return load_json('classes_parameters_expressions.json')


def get_class_parameter_ranges():
    return load_json('classes_parameters_ranges.json')

if __name__ == "__main__":
    print(get_class_hierarchy())
    print(get_class_parameter_expressions())
    print(get_class_parameter_ranges())




