import datetime
import time
import random
import functools
import numpy as np

class Timer:

    def __enter__(self):
        self.start = datetime.datetime.now()
        print('Starting {}...'.format(self.msg))
        return self

    def __exit__(self, *args):
        self.stop = datetime.datetime.now()
        self.elapsed = self.stop - self.start
        print('{} took {} seconds'.format( self.msg, self.elapsed.seconds ))


    def __init__(self, msg='operation'):
        self.msg = msg
        self.start = None
        self.end = None
        self.elapsed = None



class randomWeightedPicker:
    def __init__(self, optionsAndWeights):
        self.optionsAndWeights = optionsAndWeights

        self._cumOptionsAndWeightsCache = None

    def _calculateOptionsAndCummulativeWeights(self, optionsAndWeights):

        if self._cumOptionsAndWeightsCache is None:
            self._cumOptionsAndWeightsCache = tuple((
                    [option for option, weight in optionsAndWeights],
                    np.cumsum( [weight for option, weight in optionsAndWeights] ),
                ))

        return self._cumOptionsAndWeightsCache

    def rand(self):
        options, cumulativeWeights = self._calculateOptionsAndCummulativeWeights(self.optionsAndWeights)
        randValue = random.uniform(0, cumulativeWeights[-1])
        return next(option for option, cumulativeWeight in zip(options,cumulativeWeights) if randValue < cumulativeWeight)



def main():
    with Timer('stuff'):
        time.sleep(3)
        print('hey!')


def main():

    weights = [['a',1.5],
               ['b',0.5],
               ['c',1.0]]

    from collections import defaultdict
    from pprint import pprint

    dd = defaultdict(lambda : 0)

    rwp = randomWeightedPicker(weights)

    for i in range(100000):
        dd[rwp.rand()] += 1

    pprint(dd['a']/dd['b'])
    pprint(dd['a'] / dd['c'])





if __name__ == '__main__':
    main()