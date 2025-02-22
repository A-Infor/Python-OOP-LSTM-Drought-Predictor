import statistics
import matplotlib.pyplot as     plt
import numpy             as     np
import skill_metrics     as     sm
from   scipy.stats       import norm

class Plotter:
    
    def __init__(self, dataset):
        self.dataset              = dataset
        self.monthValues          = self.dataset.get_months()
        self.speiValues           = self.dataset.get_spei()
        self.speiNormalizedValues = self.dataset.get_spei_normalized()

    def plotModelPlots(self               , spei_dict             ,
                       dataTrueValues_dict, predictValues_dict    ,
                                            monthForPredicted_dict,
                       has_trained        , history               ):
        
        split_position = len(spei_dict['Train'])
        
        # self.showTaylorDiagrams(metrics_df)
        # self.showResidualPlots (dataTrueValues_dict, predictValues_dict)
        # self.showR2ScatterPlots(dataTrueValues_dict, predictValues_dict)
        
        self.showSpeiData(spei_dict['Test'], split_position)
        
        if not has_trained:
            self.drawModelLineGraph(history, None, self.dataset.city_name)
            self.showSpeiTest(spei_dict['Test'], split_position)
            
        self.showPredictionResults      (dataTrueValues_dict, predictValues_dict, monthForPredicted_dict)
        self.showPredictionsDistribution(dataTrueValues_dict, predictValues_dict)

    def drawModelLineGraph(self, history, city_cluster_name, city_for_training): #, showImages):
        
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
        
        axs[0, 0].plot(history.history['mae'] , 'tab:blue')
        axs[0, 0].set_title('MAE')
        axs[0, 0].legend(['loss'])
        
        axs[0, 1].plot(history.history['rmse'], 'tab:orange')
        axs[0, 1].set_title('RMSE')
        axs[0, 1].legend(['loss'])
        
        axs[1, 0].plot(history.history['mse'] , 'tab:green')
        axs[1, 0].set_title('MSE')
        axs[1, 0].legend(['loss'])
        
        axs[1, 1].plot(history.history['r2']  , 'tab:red')
        axs[1, 1].set_title('R²')
        axs[1, 1].legend(['explanation power'])
        
        for ax in axs[1]: # axs[1] = 2nd row
            ax.set(xlabel='Epochs (training)')
        
        plt.suptitle(f'Model {city_for_training}')
    
        # if(showImages):
            # plt.show()
        
        # saveFig(plt, 'Line Graph.', city_cluster_name, city_for_training)
        # plt.close()

    def showSpeiData(self, test_data, split):
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.monthValues,self.speiValues,label='SPEI Original')
        plt.xlabel('Ano')
        plt.ylabel('SPEI')
        plt.title(f'SPEI Data - {self.dataset.city_name}')
        plt.legend()
    
        plt.subplot(2,1,2)
        plt.plot(self.monthValues,self.speiNormalizedValues,label='Parcela de Treinamento')
        plt.xlabel('Ano')
        plt.ylabel('SPEI (Normalizado)')
        plt.title(f'{self.dataset.city_name}')
        plt.plot(self.monthValues[split:],test_data,'k',label='Parcela de Teste')
        plt.legend()
        plt.show()
    
    def showSpeiTest(self, test_data, split):
        y1positive = np.array(self.speiValues)>=0
        y1negative = np.array(self.speiValues)<=0
    
        plt.figure()
        plt.fill_between(self.monthValues, self.speiValues,y2=0,where=y1positive,
        color='green',alpha=0.5,interpolate=False, label='índices SPEI positivos')
        plt.fill_between(self.monthValues, self.speiValues,y2=0,where=y1negative,
        color='red',alpha=0.5,interpolate=False, label='índices SPEI negativos')
        plt.xlabel('Ano')
        plt.ylabel('SPEI')        
        plt.title(f'SPEI Data - {self.dataset.city_name}')
        plt.legend()
        plt.show()
        
    def showPredictionResults(self, dataTrueValues_dict, predictValues_dict, monthsForPredicted_dict):
        trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
        predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])
    
        reshapedMonth = np.append(monthsForPredicted_dict['Train'], monthsForPredicted_dict['Test'])
    
        speiMaxValue = np.max(self.speiValues)
        speiMinValue = np.min(self.speiValues)
    
        trueValues_denormalized  = (trueValues  * (speiMaxValue - speiMinValue) + speiMinValue)
        predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)
    
        plt.figure()
        plt.plot(reshapedMonth,  trueValues_denormalized)
        plt.plot(reshapedMonth, predictions_denormalized)
        plt.axvline(monthsForPredicted_dict['Train'][-1][-1], color='r')
        plt.legend(['Verdadeiros', 'Previstos'])
        plt.xlabel('Data')
        plt.ylabel('SPEI')
        plt.title(f'Valores verdadeiros e previstos para o final das séries. - {self.dataset.city_name}')
        plt.show()
    
    def showPredictionsDistribution(self, dataTrueValues_dict, predictValues_dict):
        trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
        predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])
    
        speiMaxValue = np.max(self.speiValues)
        speiMinValue = np.min(self.speiValues)
    
        trueValues_denormalized  = (trueValues  * (speiMaxValue - speiMinValue) + speiMinValue)
        predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)
    
        plt.figure()
        plt.scatter(x=trueValues_denormalized, y=predictions_denormalized, color=['white'], marker='^', edgecolors='black')
        plt.xlabel('SPEI Verdadeiros')
        plt.ylabel('SPEI Previstos')
        plt.title(f'{self.dataset.city_name}')
        plt.axline((0, 0), slope=1)
        plt.show()

    def define_box_properties(self, plot_name, color_code, label):
        	for k, v in plot_name.items():
        		plt.setp(plot_name.get(k), color=color_code)
        		
        	# use plot function to draw a small line to name the legend.
        	plt.plot([], c=color_code, label=label)
        	plt.legend()
    
    def drawMetricsBoxPlots(self, metrics_df, showImages):   
        # Creation of the empty dictionary:
        list_of_metrics_names = ['MAE', 'RMSE', 'MSE']
        list_of_metrics_types = ['Treinamento', 'Validação']
        list_of_models_names  = metrics_df['Municipio Treinado'].unique()
        
        metrics_dict = dict.fromkeys(list_of_metrics_names)
        for metric_name in metrics_dict.keys():
            metrics_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
            
            for metric_type in metrics_dict[metric_name].keys():
                metrics_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
        
        # Filling the dictionary:
        for metric_name in list_of_metrics_names:
            for metric_type in list_of_metrics_types:
                for model_name in list_of_models_names:
                    metrics_dict[metric_name][metric_type][model_name] = metrics_df[ metrics_df['Municipio Treinado'] == model_name ][f'{metric_name} {metric_type}'].to_list()
        
        # Plotting the graphs:
        for metric_name in list_of_metrics_names:
            training_plot   = plt.boxplot(metrics_dict[metric_name]['Treinamento'].values(), positions=np.array(np.arange(len(metrics_dict[metric_name]['Treinamento' ].values())))*2.0-0.35)
            validation_plot = plt.boxplot(metrics_dict[metric_name]['Validação'  ].values(), positions=np.array(np.arange(len(metrics_dict[metric_name]['Validação'   ].values())))*2.0+0.35)
        
            # setting colors for each groups
            self.define_box_properties(training_plot  , '#D7191C', 'Training'  )
            self.define_box_properties(validation_plot, '#2C7BB6', 'Validation')
        
            # set the x label values
            plt.xticks(np.arange(0, len(metrics_dict[metric_name]['Validação'].keys()) * 2, 2), metrics_dict[metric_name]['Validação'].keys(), rotation=45)
            
            plt.title(f'Comparison of performance of different models ({metric_name})')
            plt.xlabel('Machine Learning models')
            plt.ylabel(f'{metric_name} values')
            plt.grid(axis='y', linestyle=':', color='gray', linewidth=0.7)
            
            if(showImages):
                plt.show()
    
    def drawMetricsBarPlots(self, metrics_df, showImages):
        # Creation of the empty dictionary:
        list_of_metrics_names = ['MAE', 'RMSE', 'MSE', 'R^2']
        list_of_metrics_types = ['Treinamento', 'Validação']
        list_of_models_names  = metrics_df['Municipio Treinado'].unique()
        
        metrics_averages_dict = dict.fromkeys(list_of_metrics_names)
        for metric_name in metrics_averages_dict.keys():
            metrics_averages_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
            
            for metric_type in metrics_averages_dict[metric_name].keys():
                metrics_averages_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
        
        # Filling the dictionary:
        for metric_name in list_of_metrics_names:
            for metric_type in list_of_metrics_types:
                for model_name in list_of_models_names:
                    average = statistics.mean( metrics_df[ metrics_df['Municipio Treinado'] == model_name ][f'{metric_name} {metric_type}'].to_list() )
                    metrics_averages_dict[metric_name][metric_type][model_name] = average
        
        # Plotting the graphs:
        for metric_name in list_of_metrics_names:
            Y_axis = np.arange(len(list_of_models_names)) 
            
            # 0.4: width of the bars; 0.2: distance between the groups
            plt.barh(Y_axis - 0.2, metrics_averages_dict[metric_name]['Treinamento'].values(), 0.4, label = 'Training')
            plt.barh(Y_axis + 0.2, metrics_averages_dict[metric_name]['Validação']  .values()  , 0.4, label = 'Validation')
            
            plt.yticks(Y_axis, list_of_models_names, rotation=45)
            plt.ylabel("Machine Learning models")
            plt.xlabel(f'Average {metric_name}' if metric_name != 'R^2' else 'Average R²')
            plt.title ("Comparison of performance of different models")
            plt.legend()
            
            if(showImages):
                plt.show()
    
    def define_normal_distribution(self, axis, x_values):
        mu, std    = norm.fit     (x_values)
        xmin, xmax = axis.get_xlim()
        x          = np.linspace  (xmin, xmax, 100)
        p          = norm.pdf     (x, mu, std)
        
        return x, p
    
    def drawMetricsHistograms(self, metrics_df, showImages):
        # Creation of the empty dictionary:
        list_of_metrics_names = ['MAE', 'RMSE']
        list_of_metrics_types = ['Treinamento', 'Validação']
        list_of_models_names  = metrics_df['Municipio Treinado'].unique()
        
        metrics_dict = dict.fromkeys(list_of_metrics_names)
        for metric_name in metrics_dict.keys():
            metrics_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
            
            for metric_type in metrics_dict[metric_name].keys():
                metrics_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
        
        # Filling the dictionary:
        for metric_name in list_of_metrics_names:
            for metric_type in list_of_metrics_types:
                for model_name in list_of_models_names:
                    metrics_dict[metric_name][metric_type][model_name] = metrics_df[ metrics_df['Municipio Treinado'] == model_name ][f'{metric_name} {metric_type}'].to_list()
    
        # Plotting the graphs:        
        for model_name in list_of_models_names:
            x_MAE  = [ metrics_dict['MAE' ]['Treinamento'][model_name], metrics_dict['MAE' ]['Validação'][model_name] ]
            x_RMSE = [ metrics_dict['RMSE']['Treinamento'][model_name], metrics_dict['RMSE']['Validação'][model_name] ]
        
            fig, axs = plt.subplots(nrows=1, ncols=2)
            
            axs[0].hist(x_MAE , density=True, histtype='bar', color=['red', 'green'], label=['Treinamento', 'Validação'])
            x, p = self.define_normal_distribution(axs[0], x_MAE[0])
            axs[0].plot(x, p, 'red', linewidth=2)
            x, p = self.define_normal_distribution(axs[0], x_MAE[1])
            axs[0].plot(x, p, 'green', linewidth=2)
            axs[0].set_title('MAE')
            
            axs[1].hist(x_RMSE, density=True, histtype='bar', color=['red', 'green'], label=['Treinamento', 'Validação'])
            x, p = self.define_normal_distribution(axs[1], x_RMSE[0])
            axs[1].plot(x, p, 'red', linewidth=2)
            x, p = self.define_normal_distribution(axs[1], x_RMSE[1])
            axs[1].plot(x, p, 'green', linewidth=2)
            axs[1].set_title('RMSE')
            
            for ax in axs.flat:
                ax.set(ylabel='Frequency')
        
            plt.suptitle(f'Histograms of model {model_name}')
            fig.tight_layout()
            
            if(showImages):
                plt.show()
    
    def showResidualPlots(self, true_values_dict, predicted_values_dict, city_cluster_name, city_for_training, city_for_predicting, showImages):
        residuals        = {'Train': true_values_dict['Train'] - predicted_values_dict['Train'],
                            'Test' : true_values_dict['Test' ] - predicted_values_dict['Test' ]}
        
        for training_or_testing in ['Train', 'Test']:
            plt.scatter(predicted_values_dict[training_or_testing], residuals[training_or_testing], alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot for {training_or_testing} data. Model {city_for_training} applied to {city_for_predicting}.')
            if(showImages):
                plt.show()
    
    def showR2ScatterPlots(self, true_values_dict, predicted_values_dict, city_cluster_name, city_for_training, city_for_predicting, showImages):    
        for training_or_testing in ['Train', 'Test']:
            plt.scatter(true_values_dict[training_or_testing], predicted_values_dict[training_or_testing], label = 'R²')
            
            # Generates a single line by creating `x_vals`, a sequence of 100 evenly spaced values between the min and max values in true_values
            flattened_values = np.ravel(true_values_dict[training_or_testing])
            x_vals = np.linspace(min(flattened_values), max(flattened_values), 100)
            plt.plot(x_vals, x_vals, color='red', label='x=y')  # Line will only appear once
            
            plt.title(f'R² {training_or_testing} data. Model {city_for_training} applied to {city_for_predicting}.')
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.legend()
                
            if(showImages):
                plt.show()
    
    def drawMetricsRadarPlots(self, metrics_df, showImages):
        # Creation of the empty dictionary:
        list_of_metrics_names = ['MAE', 'RMSE', 'MSE', 'R^2']
        list_of_metrics_types = ['Treinamento', 'Validação']
        list_of_models_names  = metrics_df['Municipio Treinado'].unique()
        
        metrics_averages_dict = dict.fromkeys(list_of_metrics_names)
        for metric_name in metrics_averages_dict.keys():
            metrics_averages_dict[metric_name] = dict.fromkeys(list_of_metrics_types)
            
            for metric_type in metrics_averages_dict[metric_name].keys():
                metrics_averages_dict[metric_name][metric_type] = dict.fromkeys(list_of_models_names)
        
        # Filling the dictionary:
        for metric_name in list_of_metrics_names:
            for metric_type in list_of_metrics_types:
                for model_name in list_of_models_names:
                    average = statistics.mean( metrics_df[ metrics_df['Municipio Treinado'] == model_name ][f'{metric_name} {metric_type}'].to_list() )
                    metrics_averages_dict[metric_name][metric_type][model_name] = average
        
        # Plotting the graphs:
        for metric_type in list_of_metrics_types:
            for model_name in list_of_models_names:
                values     = [ metrics_averages_dict['MAE' ][metric_type][model_name],
                               metrics_averages_dict['RMSE'][metric_type][model_name],
                               metrics_averages_dict['MSE' ][metric_type][model_name],
                               metrics_averages_dict['R^2' ][metric_type][model_name] ]
                
                # Compute angle for each category:
                angles = np.linspace(0, 2 * np.pi, len(list_of_metrics_names), endpoint=False).tolist() + [0]
                
                plt.polar (angles, values + values[:1], color='red', linewidth=1)
                plt.fill  (angles, values + values[:1], color='red', alpha=0.25)
                plt.xticks(angles[:-1], list_of_metrics_names)
                
                # To prevent the radial labels from overlapping:
                ax = plt.subplot(111, polar=True)
                ax.set_theta_offset(np.pi / 2)   # Set the offset
                ax.set_theta_direction(-1)       # Set direction to clockwise
        
                
                plt.title (f'Performance of model {model_name} ({metric_type})')
                plt.tight_layout()
                
                if(showImages):
                    plt.show()
    
    def showTaylorDiagrams(self, metrics_df, city_cluster_name, city_for_training, city_for_predicting, showImages):
        
        label =          ['Obs', 'Train', 'Test']
        sdev  = np.array([metrics_df.iloc[-1]['Desvio Padrão Obs.'             ] ,
                          metrics_df.iloc[-1]['Desvio Padrão Pred. Treinamento'] ,
                          metrics_df.iloc[-1]['Desvio Padrão Pred. Validação'  ] ])
        ccoef = np.array([1.                                                     ,
                          metrics_df.iloc[-1]['Coef. de Correlação Treinamento'] ,
                          metrics_df.iloc[-1]['Coef. de Correlação Validação'  ] ])
        rmse  = np.array([0.                                                     ,
                          metrics_df.iloc[-1]['RMSE Treinamento'               ] ,
                          metrics_df.iloc[-1]['RMSE Validação'                 ] ])
        
        # Plotting:
        ## If both are positive, 90° (2 squares), if one of them is negative, 180° (2 rectangles)
        figsize = (2*8, 2*5) if (metrics_df.iloc[-1]['Coef. de Correlação Treinamento'] > 0 and metrics_df.iloc[-1]['Coef. de Correlação Validação'] > 0) else (2*8, 2*3)
        
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
        AVAILABLE_AXES = {'a) Training': 0, 'b) Testing': 1}
        for axs_title, axs_number in AVAILABLE_AXES.items():
            ax = axs[axs_number]
            ax.set_title(axs_title, loc="left", y=1.1)
            ax.set(adjustable='box', aspect='equal')
            sm.taylor_diagram(ax, sdev, rmse, ccoef, markerLabel = label, markerLabelColor = 'r', 
                              markerLegend = 'on', markerColor = 'r',
                              styleOBS = '-', colOBS = 'r', markerobs = 'o',
                              markerSize = 6, tickRMS = [0.0, 0.05, 0.1, 0.15, 0.2],
                              tickRMSangle = 115, showlabelsRMS = 'on',
                              titleRMS = 'on', titleOBS = 'Obs')
        plt.suptitle (f'Model {city_for_training} applied to {city_for_predicting}')
        fig.tight_layout(pad = 1.5)
        
        if(showImages):
            plt.show()