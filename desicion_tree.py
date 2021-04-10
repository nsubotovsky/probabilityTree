import pandas as pd
from cut_calculators import GiniCutterCalculator, ChiSquaredCalculator
from metrics import Metrics, MetricsAggregator
from optimal_cut_selector import BestCutSelector, TopN, RandomProportional
from functools import lru_cache
from collections import namedtuple, deque, defaultdict
from utils import Timer
from viz import HeatmapVisualizer

Prediction = namedtuple('Prediction', 'value trueProb')

class _TreeNode:

    def __init__(self, level:int, targets:pd.Series, cut=None):
        self.level=level
        self.cut = cut

        self.sampleCount = targets.count()
        self.trueVals = targets.sum()
        self.falseVals = self.sampleCount - self.trueVals

        self.lessThanNode=None
        self.greaterThanOrEqualNode=None


    def __repr__(self):
        return 'TreeNode( NodeCount: {}, True={}, False={} )'.format(self.sampleCount, self.trueVals, self.falseVals)

    def nodeLevelPrediction(self):
        return Prediction( value=self.trueVals > self.falseVals, trueProb=self.trueVals / self.sampleCount )


class Tree:

    def __init__(self, max_depth=3, cut_selector=None):
        self.cutSelector = cut_selector or BestCutSelector(GiniCutterCalculator)

        # stop conditions config
        self.max_depth = max_depth
        self.min_samples_split = 2
        self.min_samples_leaf = 1

        self.rootNode = None

        self._X = None
        self._Y = None
        self._fullIndexTemplate = None
        self._logData()

        self.usedVariables = set()

    def _logData(self):
        print(' Tree data:')
        print('   cut selector: {}'.format(self.cutSelector.__class__.__name__))
        print('   max_depth: {}'.format(self.max_depth))

    def _getFullIndex(self, Y):
        'Gets an index of all TRUES, for initializing purposes'
        return Y >= -5

    def fit(self, X : pd.DataFrame, Y : pd.Series):

        self.rootNode = _TreeNode(level=0, targets=Y )
        self._recursive(self.rootNode, X, Y)


    def predict(self, X:pd.DataFrame, Y:pd.Series=None):
        predictions = [self.predictSingle( row ) for _i, row in X.iterrows()]

        if Y is None:
            return predictions

        else:
            return predictions, Metrics( predictions, list(Y) )


    def predictSingle(self, row:pd.Series):

        currNode = self.rootNode

        while currNode.cut is not None:
            cut = currNode.cut


            if row[cut.columnName] < cut.cutThreshold:
                currNode = currNode.lessThanNode

            else:
                currNode = currNode.greaterThanOrEqualNode

        return currNode.nodeLevelPrediction()


    def _recursive(self, currentNode:_TreeNode, X:pd.DataFrame, Y:pd.Series):

        result = self._doTheCut( currentNode, X, Y)
        if result is not None:
            cut, (lessThanX, lessThanY), (greaterThanOrEqualX, greaterThanOrEqualY) = result

            currentNode.cut = cut
            currentNode.lessThanNode = _TreeNode( currentNode.level+1, lessThanY )
            currentNode.greaterThanOrEqualNode = _TreeNode(currentNode.level + 1, greaterThanOrEqualY)

            self._recursive( currentNode.lessThanNode, lessThanX, lessThanY )
            self._recursive( currentNode.greaterThanOrEqualNode, greaterThanOrEqualX, greaterThanOrEqualY )

    def _doTheCut(self, currentNode:_TreeNode, X:pd.DataFrame, Y:pd.Series):

        # stop condition checks:

        # depth check
        if currentNode.level >= self.max_depth-1:
            return

        # check leaf is big enough to split
        if currentNode.sampleCount < self.min_samples_split:
            return

        # check if we have variability in the current node
        if currentNode.falseVals == 0 or currentNode.trueVals == 0:
            return

        cut = self.cutSelector.findCut(X, Y)
        self.usedVariables.add(cut.columnName)
        lessThanIndexes = cut.lessThanIndexes(X)

        lessThanX, lessThanY = X[lessThanIndexes], Y[lessThanIndexes]
        greaterThanOrEqualX, greaterThanOrEqualY = X[~lessThanIndexes], Y[~lessThanIndexes]


        # check that split generated leaves with adequate number of leaves
        if lessThanIndexes.sum() < self.min_samples_leaf or greaterThanOrEqualY.count() < self.min_samples_leaf:
            return

        return cut, (lessThanX, lessThanY), (greaterThanOrEqualX, greaterThanOrEqualY)

    def calculateAllCuts(self):

        allCutsPerColumn = defaultdict(set)

        nodesToInspect = deque()
        nodesToInspect.append( self.rootNode )

        while nodesToInspect:
            currNode = nodesToInspect.popleft()
            if currNode.cut:
                allCutsPerColumn[currNode.cut.columnName].add(currNode.cut.cutThreshold)
                nodesToInspect.append( currNode.lessThanNode )
                nodesToInspect.append(currNode.greaterThanOrEqualNode)

        return { k:sorted(v) for k,v in allCutsPerColumn.items() }



def discretizeDf(df, classColumn='class', q=31):
    for column in df.columns:

        if column == classColumn:
            continue

        df[column] = pd.qcut(df[column], q)


    return df

def main():
    from data_holder import DataHolder, DataClass
    from synthetic_samples import CircleUniformVarianceDataGenerator, CheckersDataGenerator
    from synthetic_data.sphereSyntheticGenerator import SphereSyntheticGenerator


    max_depth = 6
    repeats = 10

    datas = []
    with Timer('Generating random data'):
        for _ in range(repeats):
            #sampleDataGenerator = CheckersDataGenerator(squares=3, noise=0.6)
            #sampleDataGenerator = CircleUniformVarianceDataGenerator(radiusThreshold=0.3, dimensions=2, noise=0.1)
            #data = DataHolder(sampleDataGenerator.generate(50))

            #sampleDataGenerator = CircleUniformVarianceDataGenerator(radiusThreshold=1.6, dimensions=30, noise=0.05)
            #sampleDataGenerator = CircleUniformVarianceDataGenerator(radiusThreshold=2.4, dimensions=30, noise=0.1,noiseDimensionModulator=(1, 5))
            sampleDataGenerator = SphereSyntheticGenerator(ratio=1.2, dimensions=50, noise=0.005,noiseDimensionModulator=(1, 10))

            sampleData = sampleDataGenerator.generate(10000)
            print('sampledata ratio: {}'.format(sampleData['class'].sum()/sampleData['class'].count()))
            sampleData = discretizeDf( sampleData, classColumn='class')
            data = DataHolder( sampleData, classColumn='class' )
            #data = DataHolder( discretizeDf(sampleData), classColumn='class' )
            datas.append(data)


    for cutSelector in [
        # BestCutSelector(ChiSquaredCalculator),
        # BestCutSelector(GiniCutterCalculator),
        RandomProportional(ChiSquaredCalculator),
        TopN(ChiSquaredCalculator, 3),
    ]:

        print('==========={}[{}]============'.format(cutSelector.__class__.__name__, cutSelector._cutScoreCalculator.__name__))

        combinedMetrics = MetricsAggregator()
        for data in datas:

            with Timer('Fitting tree'):
                tree = Tree(max_depth=max_depth, cut_selector=cutSelector)
                tree.fit( *data.train.asTuples() )

            with Timer('Predicting'):
                preds, metrics = tree.predict( *data.test.asTuples() )
                print( metrics )
                combinedMetrics += metrics

            # with Timer('Visualizing'):
            #     HeatmapVisualizer.plot(
            #         tree=tree,
            #         df=data.train.asSingleDf(),
            #         xLimits=(-0.5,0.5),
            #         yLimits=(-0.5,0.5),
            #     )

        print('=== AVG:=== {}'.format(combinedMetrics))


    print('done!')




if __name__ == '__main__':
    main()
