from  classes import Dataset, NeuralNetwork, Plotter, PerformanceEvaluator
import pandas     as pd
###############################################################################
# CITY CLUSTER 1  :
## Central city   : RIO PARDO DE MINAS
rio_pardo_de_mg_dataset = Dataset       ('Rio Pardo de Minas', 'Rio Pardo de Minas', './Data/cluster RIO PARDO DE MINAS/', 'RIO PARDO DE MINAS.xlsx')
rio_pardo_de_mg_plotter = Plotter       (rio_pardo_de_mg_dataset)
rio_pardo_de_mg_model   = NeuralNetwork ('config.json', rio_pardo_de_mg_dataset, rio_pardo_de_mg_plotter)

rio_pardo_de_mg_central_metrics , _ = rio_pardo_de_mg_model.use_neural_network ()
rio_pardo_de_mg_plotter.plotMetricsPlots(rio_pardo_de_mg_central_metrics)

## Bordering city 1: MONTEZUMA
montezuma_dataset       = Dataset       ('Montezuma', 'Rio Pardo de Minas', './Data/cluster RIO PARDO DE MINAS/', 'MONTEZUMA.xlsx')
montezuma_plotter       = Plotter       (montezuma_dataset)

_ , rio_pardo_de_mg_bordering_metrics = rio_pardo_de_mg_model.use_neural_network (dataset=montezuma_dataset, plotter=montezuma_plotter)
montezuma_plotter.plotMetricsPlots(rio_pardo_de_mg_bordering_metrics)

## Bordering city 2: FRUTA DE LEITE
fruta_de_leite_dataset  = Dataset       ('Fruta de Leite', 'Rio Pardo de Minas', './Data/cluster RIO PARDO DE MINAS/', 'FRUTA DE LEITE.xlsx')
fruta_de_leite_plotter  = Plotter       (fruta_de_leite_dataset)

_ , rio_pardo_de_mg_bordering_metrics = rio_pardo_de_mg_model.use_neural_network (dataset=fruta_de_leite_dataset, plotter=fruta_de_leite_plotter)
fruta_de_leite_plotter.plotMetricsPlots(rio_pardo_de_mg_bordering_metrics)
###############################################################################
# CITY CLUSTER 2  :
## Central city   : SÃO FRANCISCO
sao_francisco_dataset = Dataset         ('São Francisco', 'São Francisco', './Data/cluster SÃO FRANCISCO/', 'SÃO FRANCISCO.xlsx')
sao_francisco_plotter = Plotter         (sao_francisco_dataset)
sao_francisco_model   = NeuralNetwork   ('config.json', sao_francisco_dataset, sao_francisco_plotter)

sao_francisco_central_metrics , _ = sao_francisco_model.use_neural_network ()
sao_francisco_plotter.plotMetricsPlots(sao_francisco_central_metrics)
# TypeError: tuple indices must be integers or slices, not str

## Bordering city 1: PINTÓPOLIS
pintopolis_dataset    = Dataset         ('Pintópolis', 'São Francisco', './Data/cluster SÃO FRANCISCO/', 'PINTÓPOLIS.xlsx')
pintopolis_plotter    = Plotter         (pintopolis_dataset)

_ , sao_francisco_bordering_metrics = sao_francisco_model.use_neural_network (dataset=pintopolis_dataset, plotter=pintopolis_plotter)
pintopolis_plotter.plotMetricsPlots     (sao_francisco_bordering_metrics)

## Bordering city 2: JAPONVAR
japonvar_dataset    = Dataset           ('Japonvar', 'São Francisco', './Data/cluster SÃO FRANCISCO/', 'JAPONVAR.xlsx')
japonvar_plotter    = Plotter           (japonvar_dataset)

_ , sao_francisco_bordering_metrics = sao_francisco_model.use_neural_network (dataset=japonvar_dataset, plotter=japonvar_plotter)
japonvar_plotter.plotMetricsPlots       (sao_francisco_bordering_metrics)
###############################################################################
# METRICS RESULTS JOINING:
metrics_central   = pd.concat([rio_pardo_de_mg_central_metrics  , sao_francisco_central_metrics  ], ignore_index=True)
metrics_bordering = pd.concat([rio_pardo_de_mg_bordering_metrics, sao_francisco_bordering_metrics], ignore_index=True)
metrics_total     = pd.concat([metrics_central                  , metrics_bordering              ], ignore_index=True)

# pintopolis_plotter.drawMetricsBoxPlots(metrics_total)
# pintopolis_plotter.drawMetricsBarPlots(metrics_total)
###############################################################################