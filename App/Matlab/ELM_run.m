Data = csvread('../Data/csv/forest_base.csv', 1, 0);
%Data = csvread('../Data/csv/match.csv', 1, 0);
%Data = csvread('../Data/csv/match_results.csv', 1, 0);
network = ELM(Data, 80);
network.addNeurons('exp(-x^2)', 100);
network.train();
res = network.predict();
exact = network.exactCompare(network.testT, res)
meanDistance = network.meanDistanceCompare(network.testT, res)