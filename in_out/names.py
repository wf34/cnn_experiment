import os

data_types = ['X', 'Y_uscore', 'Y_actions']

def _find_2nd(string, token):
    return string.find(token, string.find(token) + 1)

def convert_replay_code_to_filename(code, data_type):
    assert data_type in data_types
    return code + '_' + data_type  + '.pickle'

def convert_filename_to_replay_code(filename, data_type):
    assert data_type in data_types
    filename = os.path.basename(filename)
    return filename[:_find_2nd(filename, '_')]
