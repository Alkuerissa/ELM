x = 50:50:1000;
s = zeros(1, length(x));
stime = zeros(1, length(x));
for i = 1:length(x)
    network = ELM(Data, 80);
    network.addNeurons('1/(1+exp(-x))', x(i));
    tic;
    network.train();
    stime(i)=toc;
    res = network.predict();
    exact = network.exactCompare(network.testT, res);
    s(i) = exact;
end

t = zeros(1, length(x));
ttime = zeros(1, length(x));
for i = 1:length(x)
    network = ELM(Data, 80);
    network.addNeurons('tanh(x)', x(i));
    tic;
    network.train();
    ttime(i) = toc;
    res = network.predict();
    exact = network.exactCompare(network.testT, res);
    t(i) = exact;
end

hold on
grid minor
xlabel('Liczba neuronów');
ylabel('Czas uczenia [s]');
plot(x, stime, '-o');
plot(x, ttime, '-o');
legend('sigmoid', 'tanh');
hold off

