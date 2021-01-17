from abc import ABC, abstractmethod
import pandas as pd
import math


class BaseCutterCalculator(ABC):

    def __init__(self, series:pd.Series, target:pd.Series):
        self._series = series
        self._target = target
        self._seriesCount = self._series.count()


    def _isTrivialCut(self, series_a:pd.Series, series_b:pd.Series) -> bool:
        'Trivial cuts are cuts which leave all items in only 1 branch'
        return any( len(s) ==0 for s in (series_a, series_b) )


    def calculateCutsGains(self) -> dict:
        results = {}

        parent_score = self._getSeriesScore(self._target)

        for cutLimit in self._series.unique():
            indexes = self._series < cutLimit

            left_series = self._target[indexes]
            right_series = self._target[~indexes]

            if self._isTrivialCut(left_series, right_series):
                continue

            left_score = self._getSeriesScore(left_series)
            right_score = self._getSeriesScore(right_series)

            cut_score = ( left_score * left_series.count() + right_score * right_series.count() ) / self._seriesCount

            results[cutLimit] = parent_score - cut_score

        return results


    @abstractmethod
    def _getSeriesScore(self, series:pd.Series) -> float:
        pass

    def _getTrueAndFalseRatios(self, series:pd.Series) -> (float, float):
        totalCount = series.count()

        trueCount = series.sum()
        falseCount = totalCount - trueCount

        trueRatio = trueCount / totalCount
        falseRatio = falseCount / totalCount

        return trueRatio, falseRatio



class GiniCutterCalculator(BaseCutterCalculator):

    def _getSeriesScore(self, tragetSeries:pd.Series) -> float:
        trueRatio, falseRatio = self._getTrueAndFalseRatios(tragetSeries)
        return 1 - trueRatio ** 2 - falseRatio ** 2


class EntropyCutterCalculator(BaseCutterCalculator):

    def _getSeriesScore(self, tragetSeries: pd.Series) -> float:
        trueRatio, falseRatio = self._getTrueAndFalseRatios(tragetSeries)
        return -(self._getPartScore(trueRatio) + self._getPartScore(falseRatio) )

    def _getPartScore(self, ratio:float) -> float:
        if ratio == 0.0:
            return 0.0

        return ratio * math.log2(ratio)



def main():

    s = pd.Series([1,2,1,33,4,5,7,8,1,2,4,5,3,6,9,7,5,1])
    t = s > 5


    from pprint import pprint
    scores = EntropyCutterCalculator(s,t).calculateCutsGains()
    pprint(scores)



if __name__ == '__main__':
    main()