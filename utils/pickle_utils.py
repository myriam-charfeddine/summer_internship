import os
import pickle

def save_pickle(filename, data, directory='pickle_files'):
    path = os.path.join(directory, filename)
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(filename, directory='pickle_files'):
    path = os.path.join(directory, filename)
    with open(path, 'rb') as file:
        return pickle.load(file)