from  classes import Cluster, Dataset, NeuralNetwork, Plotter#, PerformanceEvaluator
import pandas     as pd
###############################################################################
# CITY CLUSTER 1  :
rio_pardo_de_mg_dataset = Dataset ('Rio Pardo de Minas', 'RIO PARDO DE MINAS')
montezuma_dataset       = Dataset ('Montezuma'         , 'RIO PARDO DE MINAS')
fruta_de_leite_dataset  = Dataset ('Fruta de Leite'    , 'RIO PARDO DE MINAS')

rio_pardo_de_mg_cities  = [rio_pardo_de_mg_dataset, montezuma_dataset, fruta_de_leite_dataset]

rio_pardo_de_mg_cluster = Cluster ('Rio Pardo de Minas', rio_pardo_de_mg_cities)
###############################################################################
# CITY CLUSTER 2  :
sao_francisco_dataset   = Dataset ('São Francisco'     , 'SÃO FRANCISCO'    )
pintopolis_dataset      = Dataset ('Pintópolis'        , 'SÃO FRANCISCO'    )
japonvar_dataset        = Dataset ('Japonvar'          , 'SÃO FRANCISCO'    )

sao_francisco_cities    = [sao_francisco_dataset, pintopolis_dataset, japonvar_dataset]

sao_francisco_cluster   = Cluster ('São Francisco'     , sao_francisco_cities  )
###############################################################################
# CITY CLUSTER 1  :
rio_pardo_de_mg_plotter = Plotter       (rio_pardo_de_mg_dataset)
rio_pardo_de_mg_model   = NeuralNetwork ('config.json', rio_pardo_de_mg_dataset, rio_pardo_de_mg_plotter)
rio_pardo_de_mg_central_metrics , _   = rio_pardo_de_mg_model.use_neural_network ()
# Disabled, as these are not going to be used on Anderson's masters dissertation:
# rio_pardo_de_mg_plotter.plotMetricsPlots(rio_pardo_de_mg_central_metrics)

montezuma_plotter       = Plotter       (montezuma_dataset)
_ , rio_pardo_de_mg_bordering_metrics = rio_pardo_de_mg_model.use_neural_network (dataset=montezuma_dataset, plotter=montezuma_plotter)
# Disabled, as these are not going to be used on Anderson's masters dissertation:
# montezuma_plotter.plotMetricsPlots(rio_pardo_de_mg_bordering_metrics)

fruta_de_leite_plotter  = Plotter       (fruta_de_leite_dataset)
_ , rio_pardo_de_mg_bordering_metrics = rio_pardo_de_mg_model.use_neural_network (dataset=fruta_de_leite_dataset, plotter=fruta_de_leite_plotter)
# Disabled, as these are not going to be used on Anderson's masters dissertation:
# fruta_de_leite_plotter.plotMetricsPlots(rio_pardo_de_mg_bordering_metrics)
###############################################################################
# CITY CLUSTER 2  :
sao_francisco_plotter = Plotter         (sao_francisco_dataset)
sao_francisco_model   = NeuralNetwork   ('config.json', sao_francisco_dataset, sao_francisco_plotter)
sao_francisco_central_metrics , _   = sao_francisco_model.use_neural_network ()
# Disabled, as these are not going to be used on Anderson's masters dissertation:
# sao_francisco_plotter.plotMetricsPlots(sao_francisco_central_metrics)

pintopolis_plotter    = Plotter         (pintopolis_dataset)
_ , sao_francisco_bordering_metrics = sao_francisco_model.use_neural_network (dataset=pintopolis_dataset, plotter=pintopolis_plotter)
# Disabled, as these are not going to be used on Anderson's masters dissertation:
# pintopolis_plotter.plotMetricsPlots     (sao_francisco_bordering_metrics)

japonvar_plotter    = Plotter           (japonvar_dataset)
_ , sao_francisco_bordering_metrics = sao_francisco_model.use_neural_network (dataset=japonvar_dataset, plotter=japonvar_plotter)
# Disabled, as these are not going to be used on Anderson's masters dissertation:
# japonvar_plotter.plotMetricsPlots       (sao_francisco_bordering_metrics)
###############################################################################
# METRICS RESULTS JOINING:
metrics_central   = pd.concat([rio_pardo_de_mg_central_metrics  , sao_francisco_central_metrics  ], ignore_index=True)
metrics_bordering = pd.concat([rio_pardo_de_mg_bordering_metrics, sao_francisco_bordering_metrics], ignore_index=True)

# metrics_total     = pd.concat([metrics_central                  , metrics_bordering              ], ignore_index=True)
# pintopolis_plotter.drawMetricsBoxPlots(metrics_total)
# pintopolis_plotter.drawMetricsBarPlots(metrics_total)
###############################################################################