from cut_calculators import BaseCutterCalculator, GiniCutterCalculator
from collections import namedtuple
import pandas as pd
from abc import ABC, abstractmethod
import random
from utils import randomWeightedPicker


class Cut:

    def __init__(self, columnName:str, cutThreshold:float, gain:float):
        self.columnName = columnName
        self.cutThreshold = cutThreshold
        self.gain = gain

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Cut(columnName='{}', cutThreshold={}, gain={})".format(self.columnName, repr(self.cutThreshold), self.gain)

    def lessThanIndexes(self, dataFrame:pd.DataFrame) -> pd.Series:
        return dataFrame[self.columnName] < self.cutThreshold





class CutSelector(ABC):

    def __init__(self, cutCalculator):
        self._cutScoreCalculator = cutCalculator


    def _calculateAllCuts(self, dataFrame:pd.DataFrame, classSeries:pd.Series) -> list:
        cuts = []
        for columnName in dataFrame.columns:
            cutGainsForColumn = self._cutScoreCalculator( dataFrame[columnName], classSeries ).calculateCutsGains()

            columnCutGains = [Cut(columnName, cut, gain) for cut, gain in cutGainsForColumn.items()]
            cuts.extend(columnCutGains)

        return cuts

    @abstractmethod
    def findCut(self, dataFrame:pd.DataFrame, classSeries:pd.Series) -> Cut:
        raise NotImplementedError()


class BestCutSelector(CutSelector):

    def findCut(self, dataFrame:pd.DataFrame, classSeries:pd.Series) -> Cut:
        allCuts = self._calculateAllCuts(dataFrame, classSeries)
        return max(allCuts, key=lambda x:x.gain)


class TopN(CutSelector):

    def __init__(self, cutCalculator, topCutsCount):
        super().__init__( cutCalculator )
        self.topCutsCount = topCutsCount

    def findCut(self, dataFrame:pd.DataFrame, classSeries:pd.Series) -> Cut:
        allCuts = self._calculateAllCuts(dataFrame, classSeries)
        sortedCuts = sorted(allCuts, key=lambda x:x.gain, reverse=True)
        return random.choice( sortedCuts )


class RandomProportional(CutSelector):

    def findCut(self, dataFrame:pd.DataFrame, classSeries:pd.Series) -> Cut:
        allCuts = self._calculateAllCuts(dataFrame, classSeries)
        randomWeightPicker = randomWeightedPicker([ (cut, cut.gain) for cut in allCuts])
        randomProportionalCut = randomWeightPicker.rand()
        return randomProportionalCut



def main():
    from synthetic_samples import CircleUniformVarianceDataGenerator
    df = CircleUniformVarianceDataGenerator().generate(100)

    df['x'] = pd.qcut(df['x'], 10)
    df['y'] = pd.qcut(df['y'], 10)

    print(df.head())


    cs = RandomProportional(GiniCutterCalculator)
    cut = cs.findCut( dataFrame=df.drop('class', axis='columns'), classSeries=df['class'] )
    print(cut)

    print(cs.findCut( dataFrame=df.drop('class', axis='columns'), classSeries=df['class'] ))
    print(cs.findCut(dataFrame=df.drop('class', axis='columns'), classSeries=df['class']))
    print(cs.findCut(dataFrame=df.drop('class', axis='columns'), classSeries=df['class']))
    print(cs.findCut(dataFrame=df.drop('class', axis='columns'), classSeries=df['class']))
    print(cs.findCut(dataFrame=df.drop('class', axis='columns'), classSeries=df['class']))

    pass



if __name__ == '__main__':
    main()
