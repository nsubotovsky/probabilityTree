import statistics
from pprint import pprint

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class HeatmapVisualizer:

    @classmethod
    def _SquaresAndTestPoints( cls, tree, xColName='x', yColName='y', xLimits=None, yLimits=None ):

        allCuts = tree.calculateAllCuts()


        def _calcEdges( valuesList ):
            if not valuesList:
                return (-1, 1)

            pad = statistics.stdev( valuesList )/2
            return ( min(valuesList)-pad, max(valuesList)+pad )

        xLimits = xLimits or _calcEdges( allCuts.get(xColName, []) )
        yLimits = yLimits or _calcEdges(allCuts.get(yColName, []))


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

        return allSquaresData


    @classmethod
    def _addHeatmapToFigure(cls, ax, tree, xLimits=None, yLimits=None):

        squaresToPlot = cls._SquaresAndTestPoints(tree, xLimits=xLimits, yLimits=yLimits)
        for sd in squaresToPlot:

            ax.add_patch(Rectangle(
                xy=sd['lowerLeft'],
                width=sd['width'],
                height=sd['height'],
                linewidth=1,
                color=matplotlib.cm.coolwarm(sd['trueProb']),
                fill=True,
                alpha=1.0,
                zorder=-10,
                ))
        ax.axis('equal')

    @classmethod
    def plot(cls, tree=None, df=None, xLimits=None, yLimits=None):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if tree is not None:
            cls._addHeatmapToFigure(ax, tree, xLimits=xLimits, yLimits=yLimits)

        if df is not None:
            cls._addScatterToFigure(ax, df)

        if xLimits is not None:
            ax.set_xlim(*xLimits)

        if yLimits is not None:
            ax.set_ylim(*yLimits)


        plt.show()


    @classmethod
    def _addScatterToFigure(cls, ax, df):
        ax.scatter(
            df['x'],
            df['y'],
            c=df['class'].apply( lambda x : 'red' if x else 'blue' ),
            alpha=0.3 )