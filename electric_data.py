import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tsl.datasets import TabularDataset, DatetimeDataset


class ElectricData(DatetimeDataset):
    similarity_options = {'pearson'}

    def __init__(self, normalized=True):

        self.base_data = pd.read_csv('./data/electric/normal_data.csv')
        if normalized:
            self.base_data = pd.DataFrame(MinMaxScaler().fit_transform(self.base_data), columns=self.base_data.columns)

        shape = self.base_data.shape
        mask = np.ones((shape[0], shape[1], 1)).astype(bool)
        super().__init__(target=self.base_data,
                         mask=mask,
                         name='electric',
                         similarity_score='pearson',
                         temporal_aggregation='sum',
                         spatial_aggregation='sum',
                         default_splitting_method='temporal',
                         force_synchronization=True,
                         precision=32)


        self.target = self._parse_target(self.base_data)
        self.set_mask(mask)
        self.similarity_options = {'pearson'}

    def compute_similarity(self, similarity_method, **kwargs):
        if similarity_method == 'pearson':
            return self.base_data.corr().to_numpy()
        else:
            raise ValueError('Similarity method not supported')
