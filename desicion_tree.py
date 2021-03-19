import pandas as pd
from cut_calculators import GiniCutterCalculator
from metrics import Metrics, MetricsAggregator
from optimal_cut_selector import BestCutSelector, TopN, RandomProportional
from functools import lru_cache
from collections import namedtuple, deque, defaultdict
from utils import Timer
from viz import HeatmapVisualizer

Prediction = namedtuple('Prediction', 'value trueProb')

class _TreeNode:

    def __init__(self, level:int, nodeIndexes:pd.Series, trueVals:int, cut=None):
        self.level=level
        self.nodeIndexes = nodeIndexes
        self.cut = cut
        self.trueVals = trueVals
        self.falseVals = self.sampleCount - trueVals

        self.lessThanNode=None
        self.greaterThanOrEqualNode=None

    @property
    @lru_cache()
    def sampleCount(self) -> int:
        return self.nodeIndexes.sum()

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

        self._X = X
        self._Y = Y
        self._fullIndexTemplate = pd.Series( [False]* len(self._Y) )

        self.rootNode = _TreeNode(level=0, nodeIndexes=self._getFullIndex(Y), trueVals=Y.sum() )
        self._recursive(self.rootNode)


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


    def _completeIndex(self, currentIndexes):
        index = self._fullIndexTemplate.copy()
        index[ currentIndexes.index ] = currentIndexes
        return index

    def _recursive(self, currentNode:_TreeNode):

        result = self._doTheCut( currentNode )
        if result is not None:
            cut, lessThanIndexes, greaterThanOrEqualIndexes = result

            completeLessThanIndexes = self._completeIndex(lessThanIndexes)
            completeGreaterThanOrEqualIndexes = self._completeIndex(greaterThanOrEqualIndexes)

            currentNode.cut = cut
            currentNode.lessThanNode = _TreeNode( currentNode.level+1, completeLessThanIndexes, self._Y[completeLessThanIndexes].sum() )
            currentNode.greaterThanOrEqualNode = _TreeNode(currentNode.level + 1, completeGreaterThanOrEqualIndexes, self._Y[completeGreaterThanOrEqualIndexes].sum())

            self._recursive( currentNode.lessThanNode )
            self._recursive( currentNode.greaterThanOrEqualNode )

    def _doTheCut(self, currentNode:_TreeNode):

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

        currX = self._X[currentNode.nodeIndexes]
        currY = self._Y[currentNode.nodeIndexes]

        cut = self.cutSelector.findCut(currX, currY)
        self.usedVariables.add(cut.columnName)
        lessThanIndexes = cut.lessThanIndexes(currX)
        greaterThanOrEqualIndexes = ~lessThanIndexes

        # check that split generated leaves with adequate number of leaves
        if lessThanIndexes.sum() < self.min_samples_leaf or greaterThanOrEqualIndexes.sum() < self.min_samples_leaf:
            return

        return cut, lessThanIndexes, greaterThanOrEqualIndexes

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


def main():
    from data_holder import DataHolder, DataClass
    from synthetic_samples import CircleUniformVarianceDataGenerator, CheckersDataGenerator


    max_depth = 8
    repeats = 1

    datas = []
    with Timer('Generating random data'):
        for _ in range(repeats):
            #sampleDataGenerator = CheckersDataGenerator(squares=3, noise=0.6)
            sampleDataGenerator = CircleUniformVarianceDataGenerator(radiusThreshold=0.4, dimensions=3, noise=0.05)
            data = DataHolder( sampleDataGenerator.generate(1000), classColumn='class' )
            datas.append(data)


    for cutSelector in [
        BestCutSelector(GiniCutterCalculator),
        # RandomProportional(GiniCutterCalculator),
        # TopN(GiniCutterCalculator, 3),
    ]:

        print('==========={}============'.format(cutSelector.__class__.__name__))

        combinedMetrics = MetricsAggregator()
        for data in datas:

            with Timer('Fitting tree'):
                tree = Tree(max_depth=max_depth, cut_selector=cutSelector)
                tree.fit( *data.train.asTuples() )

            with Timer('Predicting'):
                preds, metrics = tree.predict( *data.test.asTuples() )
                print( metrics )
                combinedMetrics += metrics

            with Timer('Visualizing'):
                HeatmapVisualizer.plot(
                    tree=tree,
                    df=data.train.asSingleDf(),
                    xLimits=(-0.5,0.5),
                    yLimits=(-0.5,0.5),
                )

        print('=== AVG:=== {}'.format(combinedMetrics))


    print('done!')




if __name__ == '__main__':
    main()
