#!/usr/bin/python
# -*- coding:utf8 -*-

import json


def write_json_to_file(data, output_path, flag_verbose=False):
    with open(output_path, "w") as write_file:
        json.dump(data, write_file)
    if flag_verbose is True:
        print("Json string dumped to: %s", output_path)


def read_json_from_file(input_path):
    with open(input_path, "r") as read_file:
        python_data = json.load(read_file)
    return python_data
