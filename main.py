import pandas as pd
import math
import random
from collections import OrderedDict
from utils import Timer


from synthetic_samples import CircleUniformVarianceDataGenerator
from cut_calculators import EntropyCutterCalculator



if __name__ == '__main__':


    smp = CircleUniformVarianceDataGenerator(noise=0.1).generate(1000)

    colToTest = smp['x']
    columnCuts = pd.qcut(smp['x'], 3)


    #for colName in smp.columns:
    cp = pd.qcut(smp['x'], 8)

    print(EntropyCutterCalculator( cp, smp['class'] ).calculateCutsGains())
    # for c in cp:
    #     print(c)

