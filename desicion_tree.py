import pandas as pd
from cut_calculators import GiniCutterCalculator
from optimal_cut_selector import CutSelector, BestCutSelector, TopN
from functools import lru_cache
from collections import namedtuple, deque, defaultdict
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pprint import pprint


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


    def predict(self, X:pd.DataFrame, Y:pd.Series=None):
        return [self.predictSingle( row ) for _i, row in X.iterrows()]


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

        currX = self._X[currentNode.nodeIndexes]
        currY = self._Y[currentNode.nodeIndexes]

        cut = self.cutSelector.findCut(currX, currY)
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

    # from matplotlib import pyplot as plt
    # from matplotlib.patches import Rectangle
    #
    # # Your data
    # a = ([-0.2, 0.1, 0.7],
    #      [0.1, 0.2, 0.3])
    #
    # cols=[1,2,3]
    #
    # # Your scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    #
    # # Add rectangles
    # ax.add_patch(Rectangle(
    #     xy=(-0.5, -0.5), width=1, height=1,
    #     linewidth=1, color='blue', fill=True, alpha=0.3, zorder=-10))
    # ax.axis('equal')
    #
    # from matplotlib import cm
    # ax.scatter(a[0], a[1], c=cols, cmap = cm.coolwarm)
    #
    #
    # plt.show()
    #
    #
    # return

    from synthetic_samples import CircleUniformVarianceDataGenerator

    print('Hey!')
    df = CircleUniformVarianceDataGenerator( noise=0.05 ).generate(2000)
    print(df.head())

    #tree = Tree(max_depth=10, cut_selector=BestCutSelector(GiniCutterCalculator))
    tree = Tree( max_depth=8, cut_selector=TopN(GiniCutterCalculator, 3) )

    tree.fit( df.drop('class', axis=1), df['class'] )

    predictDf = CircleUniformVarianceDataGenerator().generate(3)

    preds = tree.predict( predictDf.drop('class', axis=1) )


    def _SquaresAndTestPoints( tree, xColName='x', yColName='y', xLimits=(-0.55,0.55), yLimits=(-0.55,0.55) ):

        allCuts = tree.calculateAllCuts()

        xCuts =  [ xLimits[0] ] + allCuts.get(xColName, []) + [ xLimits[1] ]
        yCuts =  [ yLimits[0] ] + allCuts.get(yColName, []) + [ yLimits[1] ]


        allSquaresData = []

        for ix, xCut in enumerate(xCuts[:-1]):
            for iy, yCut in enumerate(yCuts[:-1]):

                nextXcut = xCuts[ix + 1]
                nextYcut = yCuts[iy + 1]

                sd = dict(
                    lowerLeft = (xCut, yCut),
                    height =  nextYcut - yCut,
                    width = nextXcut - xCut,
                    midPoint = ( (xCut + nextXcut)/2,  (yCut + nextYcut)/2 ),
                    )

                allSquaresData.append(sd)

        testPoints = pd.DataFrame([ { xColName:sd['midPoint'][0] , yColName:sd['midPoint'][1] } for sd in allSquaresData ])

        trueProbs = [ prediction.trueProb for prediction in tree.predict( testPoints ) ]

        for sd, trueProb in zip(allSquaresData, trueProbs):
            sd['trueProb'] = trueProb


        pprint(allSquaresData)
        return allSquaresData


    squaresToPlot = _SquaresAndTestPoints(tree)




    # Your data
    # lowerlefts = [ sd['lowerleft'] for sd in squaresToPlot ]
    # cols=[ sd['trueProb'] for sd in squaresToPlot ]


    fig = plt.figure()
    ax = fig.add_subplot(111)


    # Add rectangles

    from matplotlib import cm
    for sd in squaresToPlot:

        ax.add_patch(Rectangle(
            xy=sd['lowerLeft'],
            width=sd['width'],
            height=sd['height'],
            linewidth=1,
            color=cm.coolwarm(sd['trueProb']),
            fill=True,
            alpha=1.0,
            zorder=-10,
            ))
    ax.axis('equal')
    plt.show()


    # Your scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)


    # Add rectangles

    from matplotlib import cm
    for sd in squaresToPlot:

        ax.add_patch(Rectangle(
            xy=sd['lowerLeft'],
            width=sd['width'],
            height=sd['height'],
            linewidth=1,
            color=cm.coolwarm(sd['trueProb']),
            fill=True,
            alpha=1.0,
            zorder=-10,
            ))
    ax.axis('equal')

    #
    # from matplotlib import cm
    ax.scatter(
        df['x'],
        df['y'],
        c=df['class'].apply( lambda x : 'red' if x else 'blue' ),
        alpha=0.3 )
    #
    #
    plt.show()



    plt.scatter(
        df['x'],
        df['y'],
        c=df['class'].apply( lambda x : 'red' if x else 'blue' ),
        alpha=0.3 )
    plt.show()
    #
    #
    # return


    print('done!')




if __name__ == '__main__':
    main()
