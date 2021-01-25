from cut_calculators import BaseCutterCalculator, GiniCutterCalculator
from collections import namedtuple
import pandas as pd





from pprint import pprint

class CutSelector:

    CUT_DATA = namedtuple('CUT_DATA', 'columnName cutTheshold gain')

    def __init__(self, cutCalculator:BaseCutterCalculator):
        self._cutScoreCalculator = cutCalculator


    def _calculateAllCuts(self, dataFrame:pd.DataFrame, classSeries:pd.Series) -> list:
        cuts = []
        for columnName in dataFrame.columns:
            cutGainsForColumn = self._cutScoreCalculator( dataFrame[columnName], classSeries ).calculateCutsGains()

            columnCutGains = [self.CUT_DATA(columnName, cut, gain) for cut, gain in cutGainsForColumn.items()]
            cuts.extend(columnCutGains)

        return cuts

    def findCut(self, dataFrame:pd.DataFrame, classSeries:pd.Series):
        allCuts = self._calculateAllCuts(dataFrame, classSeries)
        return max(allCuts, key=lambda x:x.gain)



def main():
    from synthetic_samples import CircleUniformVarianceDataGenerator
    df = CircleUniformVarianceDataGenerator().generate(100)

    df['x'] = pd.qcut(df['x'], 10)
    df['y'] = pd.qcut(df['y'], 10)

    print(df.head())


    cs = CutSelector( GiniCutterCalculator )
    cut = cs.findCut( dataFrame=df.drop('class', axis='columns'), classSeries=df['class'] )
    print(cut)

    pass



if __name__ == '__main__':
    main()
