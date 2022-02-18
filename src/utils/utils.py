# -*- coding: utf-8 -*-
"""
@author: zgz
"""

from functools import reduce


def args_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    # args = vars(args)
    # keys = sorted(args.keys())
    # t = Texttable()
    # t.add_rows([["Parameter", "Value"]])
    # t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    # print(t.draw())

    for arg in vars(args):
        print(arg, getattr(args, arg))


def generate_combination(l1, l2):
    res = []
    for u in l1:
        for v in l2:
            if type(u) is not list:
                u = [u]
            if type(v) is not list:
                v = [v]
            res.append(u+v)
    return res


def generate_grid_search_params(search_params):
    if len(search_params.keys()) == 1:
        return search_params.values()
    else:
        return reduce(generate_combination, search_params.values())
