import tensorflow as tf
import json
from .performance_evaluator import PerformanceEvaluator

class NeuralNetwork:

    DATA_TYPES_LIST = ['80%', '20%']

    def __init__(self, file_name, dataset, plotter):
        self.dataset        = dataset
        self.plotter        = plotter
        self.evaluator      = PerformanceEvaluator()
        
        self.configs_dict   = self._set_configs(file_name)
        self.model          = self._create_ml_model()
        self.has_trained    = False
        
        # print('Input shape:', self.model.input_shape)
        # print(self.model.summary())
    
    def _set_configs(self, file_name):
        with open(file_name) as file:
            configs_dict = json.load(file)
        
        configs_dict.update(
            {'input_shape' : (configs_dict['total_points'] - configs_dict['dense_units'], 1),
             'activation'  : ['relu', 'sigmoid'],
             'loss'        : 'mse',
             'metrics'     : ['mae',
                             tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                             'mse',
                             tf.keras.metrics.R2Score(name="r2")],
             'optimizer'   : 'adam'
            }
       )
        
        return configs_dict        

    def _create_ml_model(self):
        # print(f'Started: creation of ML model {self.dataset.city_name}')
        model = tf.keras.Sequential()
        model.add(tf.keras.Input       (shape=self.configs_dict['input_shape' ]))
        model.add(tf.keras.layers.LSTM (      self.configs_dict['hidden_units'], activation=self.configs_dict['activation'][0]))
        for _ in range(3):
            model.add(tf.keras.layers.Dense(units=self.configs_dict['dense_units'], activation=self.configs_dict['activation'][1]))
        model.compile(loss=self.configs_dict['loss'], metrics=self.configs_dict['metrics'], optimizer=self.configs_dict['optimizer'])
        # print(f'Ended: creation of ML model {self.dataset.city_name}')
        
        return model
    
    def _train_ml_model(self, dataForPrediction_dict, dataTrueValues_dict):
        print(f'\nStarted: training of ML model {self.dataset.city_name} (may take a while)')
        history = self.model.fit(
            dataForPrediction_dict['80%'],
            dataTrueValues_dict   ['80%'],
            epochs=self.configs_dict['numberOfEpochs'], batch_size=1, verbose=0)
        self.has_trained = True
        print(f'Ended  : training of ML model {self.dataset.city_name}')
        
        return history
    
    def use_neural_network(self, dataset=None, plotter=None):
        if plotter == None: plotter = self.plotter
        if dataset == None:
              dataset  = self.dataset
              is_model = True
        else: is_model = False
        
        
        (               spei_dict,             months_dict,
           dataForPrediction_dict,     dataTrueValues_dict,
         monthsForPrediction_dict, monthsForPredicted_dict) = dataset.format_data_for_model(self.configs_dict)
       
        split_position = len(spei_dict['80%'])
        if not self.has_trained:
            # flags has_trained as True:
            history        = self._train_ml_model(dataForPrediction_dict, dataTrueValues_dict)
            plotter.drawModelLineGraph           (history, self.dataset.city_cluster_name, self.dataset.city_name)
            
        print(f'Started: applying ML model {self.dataset.city_name} to city {dataset.city_name}')
        
        if is_model:
            predictValues_dict = {
                '80%' : self.model.predict(dataForPrediction_dict['80%'], verbose = 0),
                '20%' : self.model.predict(dataForPrediction_dict['20%'], verbose = 0)
                                 }
        else:
            predictValues_dict = {
                '100%': self.model.predict(dataForPrediction_dict['100%'], verbose = 0),
                '20%' : self.model.predict(dataForPrediction_dict[ '20%'], verbose = 0)
                                 }
        
        
        metrics_central, metrics_bordering = self.evaluator.evaluate(is_model, spei_dict,
            dataTrueValues_dict           , predictValues_dict     ,
            self.dataset.city_cluster_name, self.dataset.city_name , dataset.city_name  )
        
        plotter.plotDatasetPlots   (dataset, spei_dict['20%']      , split_position   ,
            self.dataset.city_cluster_name , self.dataset.city_name, dataset.city_name)
        
        if is_model:
            # metrics_central:
            self.plotter.plotModelPlots(spei_dict                     , is_model         ,
                dataTrueValues_dict           , predictValues_dict    ,
                monthsForPredicted_dict       , self.has_trained      ,
                history if not self.has_trained else None             , metrics_central  ,
                self.dataset.city_cluster_name, self.dataset.city_name, dataset.city_name)
        else:
            # metrics_bordering:
            self.plotter.plotModelPlots(spei_dict                     , is_model         ,
                dataTrueValues_dict           , predictValues_dict    ,
                monthsForPredicted_dict       , self.has_trained      ,
                history if not self.has_trained else None             , metrics_bordering,
                self.dataset.city_cluster_name, self.dataset.city_name, dataset.city_name)
        
        print(f'Ended  : applying ML model {self.dataset.city_name} to city {dataset.city_name}')
        
        return metrics_central, metrics_bordering