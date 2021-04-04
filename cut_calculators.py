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

    def _calculateCutPoints(self):
        uniques = self._series.unique()
        if self._series.dtype.name == 'category':
            return uniques

        if self._series.dtype.name in {'int64', 'float64'}:
            uniques.sort()
            mid_points = (uniques[1:] + uniques[:-1]) / 2
            return mid_points

        raise ValueError('Unexpected value type: {}; dont know how to partition'.format(self._series.dtype.name))


    def calculateCutsGains(self) -> dict:
        results = {}

        parent_score = self._getSeriesScore(self._target)

        #for cutLimit in self._series.unique():
        for cutLimit in self._calculateCutPoints():

            indexes = self._series < cutLimit

            left_series = self._target[indexes]
            right_series = self._target[~indexes]

            if self._isTrivialCut(left_series, right_series):
                continue

            left_score = self._getSeriesScore(left_series)
            right_score = self._getSeriesScore(right_series)

            cut_score = ( left_score * left_series.count() + right_score * right_series.count() ) / self._seriesCount


            # optimized version here (works for GINI only so far)
            #
            # left_score = self._getSeriesScoreMultipliedByCount(left_series)
            # right_score = self._getSeriesScoreMultipliedByCount(right_series)
            #
            # cut_score = ( left_score + right_score  ) / self._seriesCount




            results[cutLimit] = parent_score - cut_score

        return results


    @abstractmethod
    def _getSeriesScore(self, series:pd.Series) -> float:
        pass

    @abstractmethod
    def _getSeriesScoreMultipliedByCount(self, series:pd.Series) -> float:
        pass

    def _getTrueAndFalseRatios(self, series:pd.Series) -> (float, float):
        totalCount = series.count()

        trueCount = series.sum()
        falseCount = totalCount - trueCount

        trueRatio = trueCount / totalCount
        falseRatio = falseCount / totalCount

        return trueRatio, falseRatio



class GiniCutterCalculator(BaseCutterCalculator):

    def _getSeriesScore(self, targetSeries:pd.Series) -> float:
        trueRatio, falseRatio = self._getTrueAndFalseRatios(targetSeries)
        return 1 - trueRatio ** 2 - falseRatio ** 2

    def _getSeriesScoreMultipliedByCount(self, targetSeries:pd.Series) -> float:
        totalCount = targetSeries.count()
        trueCount = targetSeries.sum()
        falseCount = totalCount - trueCount
        return totalCount - (trueCount*trueCount+falseCount*falseCount)/totalCount



class EntropyCutterCalculator(BaseCutterCalculator):

    def _getSeriesScore(self, tragetSeries: pd.Series) -> float:
        trueRatio, falseRatio = self._getTrueAndFalseRatios(tragetSeries)
        return -(self._getPartScore(trueRatio) + self._getPartScore(falseRatio) )

    def _getPartScore(self, ratio:float) -> float:
        if ratio == 0.0:
            return 0.0

        return ratio * math.log2(ratio)



class ChiSquaredCalculator(BaseCutterCalculator):

    def _chiSquared(self, actual, expected, apply_yates_correction=False):
        if apply_yates_correction:
            return ((math.fabs(actual - expected)-0.5)**2/expected)

        return ((actual - expected)**2/expected)

    def _getTotalTrueFalseCount(self, series):
        total = series.count()
        true = series.sum()
        false = total-true
        return total, true, false

    def calculateCutsGains(self) -> dict:
        results = {}


        parent_total, parent_true, parent_false = self._getTotalTrueFalseCount( self._target )

        # for cutLimit in self._series.unique():
        for cutLimit in self._calculateCutPoints():

            apply_yates_correction = self._series.count() <= 10

            indexes = self._series < cutLimit

            left_series = self._target[indexes]
            right_series = self._target[~indexes]

            if self._isTrivialCut(left_series, right_series):
                continue

            left_total, left_true, left_false = self._getTotalTrueFalseCount(left_series)
            right_total, right_true, right_false = self._getTotalTrueFalseCount(right_series)


            left_true_expected = parent_true * left_total / parent_total
            left_false_expected = parent_false * left_total / parent_total

            right_true_expected = parent_true * right_total / parent_total
            right_false_expected = parent_false * right_total / parent_total

            cut_score = 0
            cut_score += self._chiSquared(left_true, left_true_expected, apply_yates_correction)
            cut_score += self._chiSquared(left_false, left_false_expected, apply_yates_correction)
            cut_score += self._chiSquared(right_true, right_true_expected, apply_yates_correction)
            cut_score += self._chiSquared(right_false, right_false_expected, apply_yates_correction)


            results[cutLimit] = cut_score

        return results

    def _getSeriesScore(self, targetSeries:pd.Series) -> float:
        trueRatio, falseRatio = self._getTrueAndFalseRatios(targetSeries)
        return 1 - trueRatio ** 2 - falseRatio ** 2

    def _getSeriesScoreMultipliedByCount(self, targetSeries:pd.Series) -> float:
        totalCount = targetSeries.count()
        trueCount = targetSeries.sum()
        falseCount = totalCount - trueCount
        return totalCount - (trueCount*trueCount+falseCount*falseCount)/totalCount


def main():

    s = pd.Series([1,2,1,33,4,5,7,8,1,2,4,5,3,6,9,7,5,1.5])
    t = s > 5


    from pprint import pprint
    scores = GiniCutterCalculator(s,t).calculateCutsGains()
    pprint(scores)



if __name__ == '__main__':
    main()