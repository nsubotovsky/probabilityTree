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


    # def _generatePointsInN_Ball(self, samplesize):
    #     '''Using Muller method as described here:
    #     http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    #     Not used since point spacing doesnt seem uniform
    #     '''
    #     us = np.random.randn(samplesize, self._dimensions)
    #     norms = np.apply_along_axis(np.linalg.norm, 1, us)
    #     radiuses = np.random.random(samplesize)**(1.0/self._dimensions)
    #     cartesian_coords = ((radiuses / norms) * us.T).T
    #     return cartesian_coords

    def _random_ball(self, samplesize, radius=1):
        '''
        code extracted from here: https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
        :param samplesize: amount of samples to generate
        :param radius: radius for points in ball
        :return: a numpy 2D array of coordinates for the points
        '''
        # First generate random directions by normalizing the length of a
        # vector of random-normal values (these distribute evenly on ball).
        random_directions = np.random.normal(size=(self._dimensions, samplesize))
        random_directions /= np.linalg.norm(random_directions, axis=0)
        # Second generate a random radius with probability proportional to
        # the surface area of a ball with a given radius.
        random_radii = np.random.random(samplesize) ** (1 / self._dimensions)
        # Return the list of random (direction & length) points.
        return radius * (random_directions * random_radii).T

    def generate(self, samplesize:int=100) -> pd.DataFrame:

        one_or_zero = np.vectorize( lambda x:max(x,0))

        def _calculateTargetClass( row ):
            noisyRow = row + self._sampleNoise()

            changeMask = np.sign(row) * np.sign(noisyRow)
            changeMask = np.apply_along_axis(one_or_zero, 0, changeMask)
            cappedNoisyRow = noisyRow * changeMask


            return np.linalg.norm(cappedNoisyRow) <= self._innerRadiusThreshold


        #samples = self._generatePointsInN_Ball(samplesize)
        samples = self._random_ball(samplesize)
        sampleClass = np.apply_along_axis(_calculateTargetClass, 1, samples)

        df = pd.DataFrame(samples, columns=['x_{}'.format(i+1) for i in range(self._dimensions)])
        df['class'] = sampleClass

        return df


def main():

    dg = SphereSyntheticGenerator(
        ratio=0.5,
        dimensions=30,
        noise=0.05,
        noiseDimensionModulator=MinMaxRange(1.0, 1.0))

    df = dg.generate(10000)

    # import matplotlib.pyplot as plt
    # plt.scatter(df['x_1'], df['x_2'], c=df['class'])
    # plt.show()
    print( df['class'].sum()/df['class'].count() )





if __name__ == '__main__':
    main()