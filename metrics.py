from functools import lru_cache


class Metrics:

    def __init__(self, predictions, actualValues):
        self.predictions = predictions
        self.actualValues = actualValues
        self.total = len(predictions)

    @property
    @lru_cache()
    def truePositives(self):
        return sum( 1 for pred, actual in zip(self.predictions, self.actualValues) if pred==actual and pred == True )

    @property
    @lru_cache()
    def predictedPositives(self):
        return sum( self.predictions )

    @property
    @lru_cache()
    def predictedNegatives(self):
        return sum( 1 for pred in self.predictions if pred == False )

    @property
    @lru_cache()
    def trueNegatives(self):
        return sum( 1 for pred, actual in zip(self.predictions, self.actualValues) if pred==actual and pred == False )


    @property
    @lru_cache()
    def precision(self):
        return self.truePositives / self.predictedPositives


    @property
    @lru_cache()
    def recall(self):
        return self.truePositives / (self.truePositives + ( self.predictedNegatives - self.trueNegatives ) )


    @property
    @lru_cache()
    def f1(self):
        return (self.precision * self.recall) /(self.precision + self.recall)

    def __str__(self):
        return 'TP={}, TN={}, Presicion={}, Recall={}, F1={}'.format( self.truePositives, self.trueNegatives, self.precision, self.recall, self.f1 )





def main():
    m = Metrics( [False, False, True, False],  [False,True,True, False] )
    print(m.precision)
    print(m.recall)
    print(m.f1)


if __name__=="__main__":
    main()