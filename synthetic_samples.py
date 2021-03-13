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

    def __init__(self, ratio:float=0.5, noise:float=0.1):
        self._ratio = ratio
        self._noise=  noise

    def _generatePointsInSquare(self, sampleSize:int) -> (float, float):
        randomPointsInSquare = np.random.random((sampleSize, 2)) - 0.5
        return [(x, y) for x, y in randomPointsInSquare]

    def _sampleNoise(self) -> float:
        return random.gauss(0, self._noise)


    def generate(self, samplesize:int=100) -> pd.DataFrame:
        # remember boundary circle's radius = sqrt( ratio / pi )

        radius_squared = self._ratio / math.pi
        def _calculateTargetClass( x, y ):
            return x*x + y*y + self._sampleNoise() <= radius_squared

        samples = self._generatePointsInSquare(samplesize)
        samplesWithClass = [ (x, y, _calculateTargetClass( x,y )) for x,y in samples ]

        return pd.DataFrame(samplesWithClass, columns=['x', 'y', 'class'])




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

        return pd.DataFrame(samplesWithClass, columns=['x', 'y', 'class'])



def main():
    import matplotlib.pyplot as plt
    #df = CircleUniformVarianceDataGenerator( ratio=0.2, noise=0.03 ).generate(1000)
    df = CheckersDataGenerator(squares=4, noise=.05).generate(1000)

    plt.scatter( df['x'],  df['y'], c=df['class'] )
    plt.show()




if __name__ == '__main__':
    main()