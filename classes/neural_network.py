import tensorflow as tf
import json

class NeuralNetwork:

    DATA_TYPES_LIST = ['Train', 'Test']

    def __init__(self, file_name, dataset, plotter):
        self.dataset        = dataset
        self.plotter        = plotter
        
        self.configs_dict   = self._set_configs(file_name)
        self.model          = self._create_ml_model()
        
        print('Input shape:', self.model.input_shape)
        print(self.model.summary())
    
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
        print('Started: creation of ML model')
        model = tf.keras.Sequential()
        model.add(tf.keras.Input       (shape=self.configs_dict['input_shape']))
        model.add(tf.keras.layers.LSTM (     self.configs_dict['hidden_units'], activation=self.configs_dict['activation'][0]))
        for dense_unit in range(3):
            model.add(tf.keras.layers.Dense(units=self.configs_dict['dense_units'], activation=self.configs_dict['activation'][1]))
        model.compile(loss=self.configs_dict['loss'], metrics=self.configs_dict['metrics'], optimizer=self.configs_dict['optimizer'])
        
        print('Ended: creation of ML model')
        
        return model

    def _make_predictions(self, train_input_sequences, speiTestForPrediction):
        predicted_spei_normalized_train = self.model.predict(train_input_sequences)
        predicted_spei_normalized_test  = self.model.predict(speiTestForPrediction)
        
        return predicted_spei_normalized_train, predicted_spei_normalized_test
        
    def use_neural_network(self, is_training, dataset=None):
        if dataset == None:
            dataset = self.dataset
        
        print('Started: applying ML model')
        # SPEI_dict  .keys() = ['Train', 'Test']
        # months_dict.keys() = ['Train', 'Test']
        spei_dict, months_dict = self.dataset.train_test_split(self.configs_dict['parcelDataTrain'])
        # IN : "(train, test)DataForPrediction": alimentar a predição da rede
        # OUT: "(train, test)DataTrueValues"   : validar se as predições da rede estão corretas
        (speiTrainForPrediction, speiTrainTrueValues,
          speiTestForPrediction,  speiTestTrueValues) =  dataset.create_input_output(spei_dict, self.configs_dict)
        # trainData_dict (to-do)
        
        # IN : "(train, test)MonthsForPrediction"    : os SPEIs foram utilizados para alimentar a predição da rede
        # OUT: "(train, test)MonthForPredictedValues": os SPEIs foram preditos
        (trainMonthsForPrediction, trainMonthForPredictedValues,
          testMonthsForPrediction, testMonthForPredictedValues ) =  dataset.create_input_output(months_dict, self.configs_dict)
        # trainMonths_dict (to-do)
       
        if is_training:
            self._train_ml_model(spei_dict, months_dict)
        
        predicted_spei_normalized_train, predicted_spei_normalized_test = self._make_predictions(speiTrainForPrediction, speiTestForPrediction)
        
        trainErrors = self._getError(speiTrainTrueValues, predicted_spei_normalized_train)
        testErrors  = self._getError(speiTestTrueValues, predicted_spei_normalized_test)
        
        self._evaluate_and_plot(is_training,
                                trainErrors                     , testErrors                     ,
                                spei_dict                                                        ,
                                speiTrainTrueValues            , speiTestTrueValues             ,
                                predicted_spei_normalized_train , predicted_spei_normalized_test ,
                                trainMonthForPredictedValues    , testMonthForPredictedValues    )
        
        print('Ended: applying ML model')
        
        return self._make_predictions(speiTrainForPrediction, speiTestForPrediction)
    
    def _evaluate_and_plot(self, is_training, trainErrors, testErrors, spei_dict, train_output_targets,  speiTestTrueValues, predicted_spei_normalized_train, predicted_spei_normalized_test, trainMonthForPredictedValues, testMonthForPredictedValues):
        self._print_errors(trainErrors, testErrors)
        
        split_position = len(spei_dict['Train'])
        self.plotter.showSpeiData(spei_dict['Test'], split_position)
        
        if is_training:
            self.plotter.showSpeiTest(spei_dict['Test'], split_position)
            
        self.plotter.showPredictionResults(train_output_targets, speiTestTrueValues, predicted_spei_normalized_train, predicted_spei_normalized_test, trainMonthForPredictedValues, testMonthForPredictedValues)
        self.plotter.showPredictionsDistribution(train_output_targets, speiTestTrueValues, predicted_spei_normalized_train, predicted_spei_normalized_test)
    
    def _train_ml_model(self, spei_dict, months_dict):
        print('Started: training of ML model (may take a while)')
        print('Started: applying ML model')
        
        train_input_sequences, train_output_targets, _, _ = self.dataset.create_input_output(spei_dict, self.configs_dict)
        
        history=self.model.fit(train_input_sequences, train_output_targets, epochs=self.configs_dict['numberOfEpochs'], batch_size=1, verbose=0)
        self.plotter.print_loss_chart(history)
        print('Ended: training of ML model')
    
    def _getError(self, actual, prediction):
        metrics = {
            'RMSE' : tf.keras.metrics.RootMeanSquaredError(),
            'MSE'  : tf.keras.metrics.MeanSquaredError    (),
            'MAE'  : tf.keras.metrics.MeanAbsoluteError   (),
            'R^2'  : tf.keras.metrics.R2Score             (class_aggregation='variance_weighted_average')
        }
    
        metrics_values = dict.fromkeys(metrics.keys())
        
        for metric_name, metric_function in metrics.items():
            metric_function.update_state(actual, prediction)
            metrics_values[metric_name] = metric_function.result().numpy()
        
        return (metrics_values)
    
    def _print_errors(self, trainErrors, testErrors):
        print("--------------Result for " +"---------------")
        print("---------------------Train-----------------------")
        print(trainErrors)
    
        print("---------------------Test------------------------")
        print(testErrors)