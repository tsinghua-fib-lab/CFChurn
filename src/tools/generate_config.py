# -*- coding: utf-8 -*-
"""
@author: zgz

"""

import yaml
from functools import reduce

input_path = 'test.yaml'
output_path = 'test'


def generate_combination(l1: list, l2: list):
    res = []
    for u in l1:
        for v in l2:
            if type(u) is not list:
                u = [u]
            if type(v) is not list:
                v = [v]
            res.append(u+v)
    return res


def generate_grid_search_params(search_params: dict):
    if len(search_params.keys()) == 1:
        return [[u] for u in list(search_params.values())[0]]
    else:
        return reduce(generate_combination, search_params.values())


def yaml_to_grid_params(input_path, output_path):
    with open(input_path, 'r') as stream:
        data = yaml.load(stream)

    for k, v in data.items():
        if type(v) is list:
            if type(v[0]) is str:
                data[k] = ['--' + str(k) + ' \'' + str(u) + '\'' for u in v]
            else:
                data[k] = ['--' + str(k) + ' ' + str(u) for u in v]
        else:
            if type(v) is str:
                data[k] = '--' + str(k) + ' \'' + str(v) + '\''
            else:
                data[k] = '--' + str(k) + ' ' + str(v)

    candidates = {u: v for u, v in data.items() if type(v) is list}
    non_candidates = [u for u, v in data.items() if type(v) is not list]
    grid_search_params = generate_grid_search_params(candidates)

    lines = []
    for params_list in grid_search_params:
        line = ''
        for u in non_candidates:
            line += data[u] + ' '
        for u in params_list:
            line += u + ' '
        line = 'python main.py ' + line.strip()
        lines.append(line + '\n\n')

    with open(output_path, 'w') as file:
        for line in lines:
            file.write(line)


if __name__ == '__main__':
    yaml_to_grid_params(input_path, output_path)
