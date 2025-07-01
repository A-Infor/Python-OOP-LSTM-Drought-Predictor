import tensorflow as tf
import numpy      as np
import pandas     as pd

class PerformanceEvaluator():
    
    def __init__(self):
        self.metrics_df = pd.DataFrame(columns=['Agrupamento', 'Municipio Treinado', 'Municipio Previsto', 'MAE 80%', 'MAE 20%', 'RMSE 80%', 'RMSE 20%', 'MSE 80%', 'MSE 20%', 'R^2 80%', 'R^2 20%', 'Desvio Padrão Obs.', 'Desvio Padrão Pred. 80%', 'Desvio Padrão Pred. 20%', 'Coef. de Correlação 80%', 'Coef. de Correlação 20%'])
        
    def evaluate          (self, has_trained   , spei_dict          ,
                           dataTrueValues_dict , predictValues_dict ,
                           city_cluster_name   , city_for_training  , city_for_predicting):
        
        errors_dict = self._print_errors(dataTrueValues_dict, predictValues_dict  ,
                                         city_for_training  , city_for_predicting , has_trained)
        self.writeErrors(errors_dict      , spei_dict        , dataTrueValues_dict, predictValues_dict,
                         city_cluster_name, city_for_training, city_for_predicting)
        
        return self.metrics_df
    
    def getError(self, actual, prediction):
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
        
        return metrics_values

    def _print_errors(self, dataTrueValues_dict, predictValues_dict, city_for_training, city_for_predicting, has_trained):
    
        match has_trained:
            case False:
                print(f'\t\t--------------Result for {city_for_training} (training)---------------')
            case True :
                print(f'\t\t--------------Result for {city_for_training} applied to {city_for_predicting}---------------')
            case _    :
                print('Error in method _print_errors of class PerformanceEvaluator: the has_trained state cannot be recognized.')
                return False
    
        # RMSE, MSE, MAE, R²:
        errors_dict = {
            '80%': self.getError(dataTrueValues_dict['80%'], predictValues_dict['80%']),
            '20%' : self.getError(dataTrueValues_dict['20%' ], predictValues_dict['20%' ])
                      }
    
        print(f"\t\t\tTRAIN: {errors_dict['80%']}")
        print(f"\t\t\tTEST : {errors_dict['20%'] }")
        
        return errors_dict

    def writeErrors(self, errors_dict  , spei_dict          ,
                    dataTrueValues_dict, predictValues_dict ,
                    city_cluster_name  , city_for_training  , city_for_predicting):
        observed_std_dev, predictions_std_dev, correlation_coefficient = self.getTaylorMetrics(spei_dict, dataTrueValues_dict, predictValues_dict)
        
        row = {
            'Agrupamento'                    : city_cluster_name                        ,
            'Municipio Treinado'             : city_for_training                        ,
            'Municipio Previsto'             : city_for_predicting                      ,
            'MAE 80%'                : errors_dict            ['80%']['MAE' ] ,
            'MAE 20%'                  : errors_dict            ['20%' ]['MAE' ] ,
            'RMSE 80%'               : errors_dict            ['80%']['RMSE'] ,
            'RMSE 20%'                 : errors_dict            ['20%' ]['RMSE'] ,
            'MSE 80%'                : errors_dict            ['80%']['MSE' ] ,
            'MSE 20%'                  : errors_dict            ['20%' ]['MSE' ] ,
            'R^2 80%'                : errors_dict            ['80%']['R^2' ] ,
            'R^2 20%'                  : errors_dict            ['20%' ]['R^2' ] ,
            'Desvio Padrão Obs.'             : observed_std_dev                         ,
            'Desvio Padrão Pred. 80%': predictions_std_dev    ['80%']         ,
            'Desvio Padrão Pred. 20%'  : predictions_std_dev    ['20%' ]         ,
            'Coef. de Correlação 80%': correlation_coefficient['80%']         ,
            'Coef. de Correlação 20%'  : correlation_coefficient['20%' ]
        }

        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([row])], ignore_index=True)

    def getTaylorMetrics(self, spei_dict, dataTrueValues_dict, predictValues_dict):    
     # Standard Deviation:
     predictions_std_dev       = {'80%': np.std(predictValues_dict['80%']),
                                  '20%' : np.std(predictValues_dict['20%' ])}
     
     combined_data             = np.concatenate([spei_dict['80%'], spei_dict['20%']])
     observed_std_dev          = np.std(combined_data)
     
     print(f"\t\t\tTRAIN: STD Dev {predictions_std_dev['80%']}")
     print(f"\t\t\tTEST : STD Dev {predictions_std_dev['20%' ]}")
     
     # Correlation Coefficient:
     correlation_coefficient  = {'80%': np.corrcoef(predictValues_dict['80%'], dataTrueValues_dict['80%'])[0, 1],
                                 '20%' : np.corrcoef(predictValues_dict['20%' ], dataTrueValues_dict['20%' ])[0, 1]}
     
     print(f"\t\t\tTRAIN: correlation {correlation_coefficient['80%']}")
     print(f"\t\t\tTEST : correlation {correlation_coefficient['20%' ]}")
     
     return observed_std_dev, predictions_std_dev, correlation_coefficient