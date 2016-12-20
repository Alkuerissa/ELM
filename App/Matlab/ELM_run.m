Data = csvread('../Data/csv/forest_base.csv', 1, 0);
network = ELM(Data, 90);
network.addNeurons('x', 50);
network.train();
res = network.predict();
network.exactCompare(network.testT, res)
network.meanDistanceCompare(network.testT, res)