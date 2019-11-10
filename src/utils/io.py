import pickle


def load(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def save(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
