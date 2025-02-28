from classes import Dataset, NeuralNetwork, Plotter, PerformanceEvaluator

rio_pardo_de_mg_dataset   = Dataset       ('Rio Pardo de Minas', 'Rio Pardo de Minas', './Data/', 'spei12_riopardodeminas.xlsx')
rio_pardo_de_mg_plotter   = Plotter       (rio_pardo_de_mg_dataset)
rio_pardo_de_mg_model     = NeuralNetwork ('config.json', rio_pardo_de_mg_dataset, rio_pardo_de_mg_plotter)

metrics_df = rio_pardo_de_mg_model.use_neural_network ()
rio_pardo_de_mg_plotter.plotMetricsPlots              (metrics_df)

francisco_sá_dataset    = Dataset ('Francisco Sá', 'Francisco Sá', './Data/', 'spei12_FranciscoSá.xlsx')
francisco_sá_plotter    = Plotter (francisco_sá_dataset)

metrics_df = rio_pardo_de_mg_model.use_neural_network (dataset=francisco_sá_dataset, plotter=francisco_sá_plotter)
francisco_sá_plotter.plotMetricsPlots                 (metrics_df)