import collections
import os
import operator
import pickle
import cnn_experiment.in_out.names as io_n


def generate_replay_codes(directory):
    for f in os.listdir(os.path.join(directory, io_n.data_types[0])):
        yield io_n.convert_filename_to_replay_code(f, io_n.data_types[0])


def get_relavant_files(code, directory):
    directories = [os.path.join(directory, dt) for dt in io_n.data_types]
    print('directories', [os.path.basename(x) for x in directories])
    relevant_files = [os.path.join(d, io_n.convert_replay_code_to_filename(
        code, os.path.basename(d))) for d in directories]
    assert all([os.path.isfile(x) for x  in relevant_files]), code
    return relevant_files


class step_with_datum(
    collections.namedtuple('step_with_datum', ['step', 'datum'])):
    """universal class to represent data from encoded replay_file"""
    #def __init__(self, *args):
    #    assert len(args) == 2
    #    self.step = args[0]
    #    self.datum = args[1]


class decoding_file_iterator():
    """Class for internal use;
       Is an iterable, which yeilds one specific `data_type` entry associated
           with each replay step in a one given replay.
    """

    def __init__(self, filepath):
        assert os.path.isfile(filepath), filepath
        self.filepath = filepath


    def __iter__(self):
        with open(self.filepath, 'rb') as the_file:
            while True:
                try:
                    x = [pickle.load(the_file) for _ in range(2)]
                    yield step_with_datum(*x)
                except EOFError:
                    break
        raise StopIteration


def get_uscore_for_the_game(replay_files):
    fit = decoding_file_iterator(replay_files[1])
    Y_uscores = [x for x in fit]
    assert len(Y_uscores) == 1
    return Y_uscores[0]


class data_iterator():
    """Is an iterable, which yeilds all data associated with each replay step
       for all replays in `input_dir`
    """
    def __init__(self, input_dir):
        """Checks that all `data_types` are available in a provided dataset

            Args:
                input_dir: path to the full dataset
        """
        self.input_dir = input_dir
        input_dirs = list(map(lambda x: os.path.join(input_dir, x),
                           io_n.data_types))
        assert all([os.path.isdir(x) for x in input_dirs]), input_dirs


    def iterate_replay(self, replay_files):
        assert isinstance(replay_files, list)
        assert len(replay_files) == 3
        Y_uscore = get_uscore_for_the_game(replay_files)
        xy_files = operator.itemgetter(*[0, -1])(replay_files)
        decoders = [decoding_file_iterator(f) for f in xy_files]
        for X, Y_actions in zip(*decoders):
            yield X, Y_uscore, Y_actions

        stop_statuses = 0
        for d in decoders:
            try:
                next(d)
            except StopIteration:
                stop_statuses += 1
        assert stop_statuses == 2


    def __iter__(self):
        for replay_code in generate_replay_codes(self.input_dir):
            replay_files = get_relavant_files(replay_code, self.input_dir)
            for X, Y_uscore, Y_actions in self.iterate_replay(replay_files):
                assert X.step == Y_actions.step  
                yield replay_code, X.step, X.datum, Y_uscore.datum, Y_actions.datum

