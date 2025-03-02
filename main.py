from classes import Dataset, NeuralNetwork, Plotter, PerformanceEvaluator

rio_pardo_de_mg_dataset   = Dataset       ('Rio Pardo de Minas', 'Rio Pardo de Minas', './Data/', 'RIO PARDO DE MINAS.xlsx')
rio_pardo_de_mg_plotter   = Plotter       (rio_pardo_de_mg_dataset)
rio_pardo_de_mg_model     = NeuralNetwork ('config.json', rio_pardo_de_mg_dataset, rio_pardo_de_mg_plotter)

metrics_df = rio_pardo_de_mg_model.use_neural_network ()
rio_pardo_de_mg_plotter.plotMetricsPlots              (metrics_df)

montezuma_dataset         = Dataset ('Montezuma', 'Rio Pardo de Minas', './Data/', 'MONTEZUMA.xlsx')
montezuma_plotter         = Plotter (montezuma_dataset)

metrics_df = rio_pardo_de_mg_model.use_neural_network (dataset=montezuma_dataset, plotter=montezuma_plotter)
montezuma_plotter.plotMetricsPlots                 (metrics_df)