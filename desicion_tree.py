import pandas as pd
from cut_calculators import GiniCutterCalculator
from optimal_cut_selector import CutSelector
from functools import lru_cache



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




class Tree:

    def __init__(self):
        self.cutSelector = CutSelector(GiniCutterCalculator)

        # stop conditions config
        self.max_depth = 3
        self.min_samples_split = 2
        self.min_samples_leaf = 1

        self.rootNode = None

        self._X = None
        self._Y = None
        self._fullIndexTemplate = None

    def _calculateBranchCut(self, X:pd.DataFrame, Y:pd.DataFrame):
        cut = self.cutSelector.findCut(X, Y)
        lessThanIndexes = cut.lessThanIndexes(X)
        greaterThanOrEqualIndexes = ~lessThanIndexes
        return _TreeNode(cut, lessThanIndexes, greaterThanOrEqualIndexes)

    def _getFullIndex(self, Y):
        'Gets an index of all TRUES, for initializing purposes'
        return Y >= -5

    def fit(self, X : pd.DataFrame, Y : pd.Series):

        self._X = X
        self._Y = Y
        self._fullIndexTemplate = pd.Series( [False]* len(self._Y) )

        self.rootNode = _TreeNode(level=0, nodeIndexes=self._getFullIndex(Y), trueVals=Y.sum() )
        self._recursive(self.rootNode)


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

        currX = self._X[currentNode.nodeIndexes]
        currY = self._Y[currentNode.nodeIndexes]

        cut = self.cutSelector.findCut(currX, currY)
        lessThanIndexes = cut.lessThanIndexes(currX)
        greaterThanOrEqualIndexes = ~lessThanIndexes

        # check that split generated leaves with adequate number of leaves
        if lessThanIndexes.sum() < self.min_samples_leaf or greaterThanOrEqualIndexes.sum() < self.min_samples_leaf:
            return

        return cut, lessThanIndexes, greaterThanOrEqualIndexes








def main():
    from synthetic_samples import CircleUniformVarianceDataGenerator

    print('Hey!')
    df = CircleUniformVarianceDataGenerator().generate(100)
    print(df.head())

    tree = Tree()

    tree.fit( df.drop('class', axis=1), df['class'] )

    print('done!')




if __name__ == '__main__':
    main()
