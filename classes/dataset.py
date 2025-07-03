import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split

class Dataset:
    
    DATA_TYPES_LIST = ['100%', '80%', '20%']
    
    def __init__(self, city_name, city_cluster_name, root_dir, xlsx):
        self.city_name         = city_name
        self.city_cluster_name = city_cluster_name
        self.df                = pd.read_excel(root_dir + xlsx, index_col=0)
        self.df.rename(columns = {'Series 1': 'SPEI Real'}, inplace=True)

    def get_months(self):
        return self.df.index.to_numpy()
    
    def get_spei(self):
        return self.df['SPEI Real'].to_numpy()
    
    def get_spei_normalized(self):
        spei = self.get_spei()
        return ( (spei - spei.min()) / (spei.max() - spei.min()) )
    
    def format_data_for_model(self, configs_dict):
        #(SPEI/months)_dict.keys() = ['80%', '20%']
        spei_dict               , months_dict             = self._train_test_split(configs_dict['parcelDataTrain'])
        
        #         IN            ,           OUT           :
        dataForPrediction_dict  , dataTrueValues_dict     =  self._create_input_output_pairs(  spei_dict, configs_dict)
        monthsForPrediction_dict, monthsForPredicted_dict =  self._create_input_output_pairs(months_dict, configs_dict)
        
        return (               spei_dict,             months_dict,
                  dataForPrediction_dict,     dataTrueValues_dict,
                monthsForPrediction_dict, monthsForPredicted_dict)
    
    def _train_test_split(self, train_size):
        
        spei_dict   = dict.fromkeys(Dataset.DATA_TYPES_LIST)
        months_dict = dict.fromkeys(Dataset.DATA_TYPES_LIST)
        
        spei_dict  ['100%'] = self.get_spei_normalized()
        months_dict['100%'] = self.get_months         ()
        
        (  spei_dict['80%'],   spei_dict['20%'],
         months_dict['80%'], months_dict['20%']) = train_test_split(spei_dict    ['100%']  ,
                                                                    months_dict  ['100%']  ,
                                                                    train_size = train_size,
                                                                    shuffle    = False     )

        return spei_dict, months_dict
    
    def _create_input_output_pairs(self, data_dict, configs_dict):
        window_gap  = configs_dict['total_points']
        dense_units = configs_dict['dense_units' ]
        
        input_dict  = dict.fromkeys(Dataset.DATA_TYPES_LIST)
        output_dict = dict.fromkeys(Dataset.DATA_TYPES_LIST)
        
        for train_or_test in Dataset.DATA_TYPES_LIST:
            # Data → sliding windows (with overlaps):
            windows = np.lib.stride_tricks.sliding_window_view(data_dict[train_or_test], window_gap)
            
            # -overlaps by selecting only every 'window_gap'-th window:
            windows = windows[::window_gap]
            
            # Last 'dense_units' elements from each window → output;
            # Remaining elements in each window            → input :
            output_dict[train_or_test] = windows[ : , -dense_units :              ]
            input_dict [train_or_test] = windows[ : ,              : -dense_units ]
            
            # +new dimension at the end of the array:
            input_dict[train_or_test] = input_dict[train_or_test][..., np.newaxis]
        
        return input_dict, output_dict