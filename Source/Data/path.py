__author__ = 'HJ'

from os.path import realpath, dirname


def get_data_path():
    """
    Get data folder path
    :return: path of data folder
    """
    cur_file_path = realpath(__file__)
    return dirname(cur_file_path)
