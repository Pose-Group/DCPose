#!/usr/bin/python
# -*- coding:utf8 -*-

import os
from .utils_natural_sort import natural_sort


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def folder_exists(folder_path):
    return os.path.exists(folder_path)


def list_immediate_childfile_paths(folder_path, ext=None, exclude=None):
    files_names = list_immediate_childfile_names(folder_path, ext, exclude)
    files_full_paths = [os.path.join(folder_path, file_name) for file_name in files_names]
    return files_full_paths


def list_immediate_childfile_names(folder_path, ext=None, exclude=None):
    files_names = [file_name for file_name in next(os.walk(folder_path))[2]]
    if ext is not None:
        if isinstance(ext, str):
            files_names = [file_name for file_name in files_names if file_name.endswith(ext)]
        elif isinstance(ext, list):
            temp_files_names = []
            for file_name in files_names:
                for ext_item in ext:
                    if file_name.endswith(ext_item):
                        temp_files_names.append(file_name)
            files_names = temp_files_names
    if exclude is not None:
        files_names = [file_name for file_name in files_names if not file_name.endswith(exclude)]
    natural_sort(files_names)
    return files_names


def list_immediate_subfolder_paths(folder_path):
    subfolder_names = list_immediate_subfolder_names(folder_path)
    subfolder_paths = [os.path.join(folder_path, subfolder_name) for subfolder_name in subfolder_names]
    return subfolder_paths


def list_immediate_subfolder_names(folder_path):
    subfolder_names = [folder_name for folder_name in os.listdir(folder_path)
                       if os.path.isdir(os.path.join(folder_path, folder_name))]
    natural_sort(subfolder_names)
    return subfolder_names
