from sklearn.model_selection import train_test_split
import pandas as pd

class DataClass:

    def __init__(self, X, Y, classColumn='class'):
        self.X = X
        self.Y = Y
        self.classColumn = classColumn

    @classmethod
    def fromDf(cls, df, classColumn):
        return DataClass(df.drop(classColumn, axis=1), df[classColumn], classColumn)

    def asTuples(self):
        return( self.X.copy(), self.Y.copy() )

    def asSingleDf(self):
        returnDf = self.X.copy()
        returnDf[self.classColumn] = self.Y
        return returnDf


class DataHolder:

    def __init__(self, df=None, classColumn='class', trainSizeRatio=0.7):
        self.dc = DataClass.fromDf(df, classColumn)
        self.classColumn = classColumn
        self.trainSizeRatio = trainSizeRatio
        self._splitTrainTest(trainSizeRatio)

    def _splitTrainTest(self, train_size):
        X_train, X_test, y_train, y_test = train_test_split(self.dc.X, self.dc.Y, train_size=self.trainSizeRatio)
        self.train = DataClass(X_train.reset_index(drop=True), y_train.reset_index(drop=True))
        self.test =  DataClass(X_test.reset_index(drop=True), y_test.reset_index(drop=True))



def main():
    import pandas as pd
    df = pd.DataFrame([{'a': 1, 'b': 2, 'c': 3},
                       {'a': 4, 'b': 5, 'c': 6},
                       {'a': 7, 'b': 8, 'c': 9}])
    dh  = DataHolder(df, classColumn='c', trainSizeRatio=0.667)
    pass


if __name__ == '__main__':
    main()