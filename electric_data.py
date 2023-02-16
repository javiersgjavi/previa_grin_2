import pandas as pd
from tsl.datasets import TabularDataset


class ElectricData(TabularDataset):
    def __init__(self, name='electric',
                 force_synchronization=True,
                 precision=32,
                 mask=None):

        self.base_data = pd.read_csv('./data/electric/normal_data.csv')
        super().__init__(name=name,
                         target=self.base_data,
                         similarity_score={'pearson'},
                         temporal_aggregation='sum',
                         spatial_aggregation='sum',
                         default_splitting_method='temporal',)

        self.precision = precision
        self.force_synchronization = force_synchronization

        self.target = self._parse_target(self.base_data)
        self.set_mask(mask)
        self.similarity_options = {'pearson'}

    def compute_similarity(self, similarity_method, **kwargs):
        if similarity_method == 'pearson':
            return self.base_data.corr().to_numpy()
        else:
            raise ValueError('Similarity method not supported')



