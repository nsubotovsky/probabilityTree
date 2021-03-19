import random
import math
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseDataGenerator(ABC):

    @abstractmethod
    def generate(self, samplesize:int=100):
        pass


class CircleUniformVarianceDataGenerator(BaseDataGenerator):

    def __init__(self, radiusThreshold:float=0.3, dimensions=2, noise:float=0.1, noiseDimensionModulator:tuple=(1.0, 1.0)):
        self._radiusThreshold = radiusThreshold
        self._dimensions = dimensions
        self._noises_per_dimension = self._calculateDimensionNoisesAmplitude( noise, noiseDimensionModulator )

    def _calculateDimensionNoisesAmplitude(self, noise, noiseDimensionModulator):
        return tuple(noise * random.uniform( *noiseDimensionModulator ) for _ in range( self._dimensions ) )

    def _generatePointsInSquare(self, sampleSize:int) -> (float, float):
        return np.random.random((sampleSize, self._dimensions)) - 0.5

    def _sampleNoise(self) -> tuple:
        return tuple(random.gauss(0, noiseInDimension) for noiseInDimension in self._noises_per_dimension)


    def generate(self, samplesize:int=100) -> pd.DataFrame:

        def _calculateTargetClass( row ):
            noisyRow = row + self._sampleNoise()
            return np.linalg.norm(noisyRow) <= self._radiusThreshold

        samples = self._generatePointsInSquare(samplesize)
        sampleClass = [ _calculateTargetClass( row ) for row in samples ]

        df = pd.DataFrame(samples, columns=['x_{}'.format(i+1) for i in range(self._dimensions)])
        df['class'] = sampleClass

        return df




class CheckersDataGenerator(BaseDataGenerator):

    def __init__(self, squares:int=2, noise:float=0.1):
        self._squares = squares
        self._noise=  noise

    def _generatePointsInSquare(self, sampleSize:int) -> (float, float):
        randomPointsInSquare = np.random.random((sampleSize, 2)) - 0.5
        return [(x, y) for x, y in randomPointsInSquare]

    def _sampleNoise(self) -> float:
        return random.gauss(0, self._noise)


    def generate(self, samplesize:int=100) -> pd.DataFrame:


        def _calculateTargetClass( x, y ):
            xAdj = (x + 0.5) * self._squares
            yAdj = (y + 0.5) * self._squares
            trueVal = bool((int(xAdj) + int(yAdj)) % 2)

            xTarget = int(xAdj) + 1 / self._squares
            yTarget = int(yAdj) + 1 / self._squares

            infDist = max( (abs( xAdj-xTarget ), abs( yAdj-yTarget )) )

            if self._sampleNoise() + ( 1 - infDist) < 0:
                return not trueVal

            else:
                return trueVal



        samples = self._generatePointsInSquare(samplesize)
        samplesWithClass = [ (x, y, _calculateTargetClass( x,y )) for x,y in samples ]

        return pd.DataFrame(samplesWithClass, columns=['x_1', 'x_2', 'class'])



def main():
    import matplotlib.pyplot as plt
    df = CircleUniformVarianceDataGenerator(radiusThreshold=0.3, noise=0.02, dimensions=3, noiseDimensionModulator=[1,5]).generate(1000)
    #df = CheckersDataGenerator(squares=4, noise=.05).generate(1000)

    plt.scatter(df['x_1'],  df['x_2'], c=df['class'] )
    plt.show()
    plt.scatter(df['x_2'], df['x_3'], c=df['class'])
    plt.show()
    plt.scatter(df['x_1'], df['x_3'], c=df['class'])
    plt.show()




if __name__ == '__main__':
    main()