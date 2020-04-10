import hashlib
import json
import os
import shutil

import numpy as np

from sklearn.model_selection import ParameterSampler

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import user_based_train_test_split
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.evaluation import sequence_mrr_score
import lenskit.datasets as d


CUDA = (os.environ.get('CUDA') is not None or
        shutil.which('nvidia-smi') is not None)

NUM_SAMPLES = 100

LEARNING_RATES = [1e-3]
LOSSES = ['pointwise']
BATCH_SIZE = [8]
EMBEDDING_DIM = [8]
N_ITER = [9]
L2 = [1e-6]


class Results:

    def __init__(self, filename):

        self._filename = filename

        open(self._filename, 'a+')

    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save(self, hyperparams, test_mrr, validation_mrr):

        result = {'test_mrr': test_mrr,
                  'validation_mrr': validation_mrr,
                  'hash': self._hash(hyperparams)}
        result.update(hyperparams)

        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best(self):

        results = sorted([x for x in self],
                         key=lambda x: -x['test_mrr'])

        if results:
            return results[0]
        else:
            return None

    def __getitem__(self, hyperparams):

        params_hash = self._hash(hyperparams)

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                if datum['hash'] == params_hash:
                    del datum['hash']
                    return datum

        raise KeyError

    def __contains__(self, x):

        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                del datum['hash']

                yield datum

def sample_lstm_hyperparameters(random_state, num):

    space = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:

        yield params


def get_k_recommendations(k=5):
    results = sorted([x for x in self], key=lambda x: -x['test_mrr'])
    return results[:k] if results else None

def get_movie_from_id(id):
    return d.ML100K('../../dataset/ml-100k/ml-100k').movies.iloc[id].title

def id_from_movie(given_movie):
    movies = d.ML100K('../../dataset/ml-100k/ml-100k').movies
    return movies.index[movies['title'] == given_movie].tolist()[0]

def evaluate_lstm_model(hyperparameters, train, test, validation, random_state):

    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='lstm',
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)
    val_mrr = sequence_mrr_score(model, validation)

    return test_mrr, val_mrr, model

def run(train, test, validation, random_state, model_type):

    results = Results('{}_results.txt'.format(model_type))

    best_result = results.best()

    eval_fnc, sample_fnc = (evaluate_lstm_model,
                                sample_lstm_hyperparameters)
    model = None
    if best_result is not None:
        print('Best {} result: {}'.format(model_type, results.best()))

    for hyperparameters in sample_fnc(random_state, NUM_SAMPLES):

        if hyperparameters in results:
            continue

        print('Evaluating {}'.format(hyperparameters))

        (test_mrr, val_mrr, model) = eval_fnc(hyperparameters,
                                       train,
                                       test,
                                       validation,
                                       random_state)

        print('Test MRR {} val MRR {}'.format(
            test_mrr.mean(), val_mrr.mean()
        ))
        test_input = [50, 176, 172, 69, 235]
        p = model.predict(test_input)
        print(np.argmax(p))
        print("I should watch ", get_movie_from_id(np.argmax(p)))

        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean())

    return results, model


if __name__ == '__main__':

    max_sequence_length = 200
    min_sequence_length = 20
    step_size = 200
    random_state = np.random.RandomState(100)

    dataset = get_movielens_dataset('100K')

    train, rest = user_based_train_test_split(dataset,
                                              random_state=random_state)
    test, validation = user_based_train_test_split(rest,
                                                   test_percentage=0.5,
                                                   random_state=random_state)
    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              min_sequence_length=min_sequence_length,
                              step_size=step_size)

   
    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length,
                            step_size=step_size)
    validation = validation.to_sequence(max_sequence_length=max_sequence_length,
                                        min_sequence_length=min_sequence_length,
                                        step_size=step_size)

    mode = 'lstm'
    
    results, model = run(train, test, validation, random_state, mode)

    # Testing for user-given movies

