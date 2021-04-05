from abc import ABC, abstractmethod
import pandas as pd

class BaseSyntheticGenerator(ABC):

    @abstractmethod
    def generate(self, sampleSize:int=100) -> pd.DataFrame:
        pass
