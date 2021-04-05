from synthetic_data.baseSyntheticGenerator import BaseSyntheticGenerator
import numpy as np
import pandas as pd
from collections import namedtuple
import random


MinMaxRange = namedtuple('MinMaxRange', 'min max')

class SphereSyntheticGenerator(BaseSyntheticGenerator):


    def __init__(self, ratio:float=0.5, dimensions=2, noise:float=0.1, noiseDimensionModulator:MinMaxRange=MinMaxRange(1.0, 1.0)):
        self._ratio = ratio
        self._dimensions = dimensions

        self._noises_per_dimension = self._calculateDimensionNoisesAmplitude( noise, noiseDimensionModulator )
        self._innerRadiusThreshold = ratio ** (1/dimensions)


    def _calculateDimensionNoisesAmplitude(self, noise, noiseDimensionModulator):
        return tuple(noise * random.uniform( *noiseDimensionModulator ) for _ in range( self._dimensions ) )

    def _sampleNoise(self) -> tuple:
        return tuple(random.gauss(0, noiseInDimension) for noiseInDimension in self._noises_per_dimension)


    def _generatePointsInN_Ball(self, samplesize):
        '''Using Muller method as described here:
        http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/'''
        us = np.random.randn(samplesize, self._dimensions)
        norms = np.apply_along_axis(np.linalg.norm, 1, us)
        radiuses = np.random.random(samplesize)**(1.0/self._dimensions)
        cartesian_coords = ((radiuses / norms) * us.T).T
        return cartesian_coords

    def generate(self, samplesize:int=100) -> pd.DataFrame:

        def _calculateTargetClass( row ):
            noisyRow = row + self._sampleNoise()
            return np.linalg.norm(noisyRow) <= self._innerRadiusThreshold


        samples = self._generatePointsInN_Ball(samplesize)
        sampleClass = np.apply_along_axis(_calculateTargetClass, 1, samples)

        df = pd.DataFrame(samples, columns=['x_{}'.format(i+1) for i in range(self._dimensions)])
        df['class'] = sampleClass

        return df


def main():

    dg = SphereSyntheticGenerator(
        ratio=0.33,
        dimensions=20,
        noise=0.0,
        noiseDimensionModulator=MinMaxRange(1.0, 10.0))

    df = dg.generate(10000)

    # import matplotlib.pyplot as plt
    # plt.scatter(df['x_1'], df['x_2'], c=df['class'])
    # plt.show()
    print( df['class'].sum()/df['class'].count() )



if __name__ == '__main__':
    main()