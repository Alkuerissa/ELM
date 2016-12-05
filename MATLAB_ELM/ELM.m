classdef ELM
    properties
        X
        T
        Neurons
        NeuronsCount
        W
        B
        H
        Beta
    end
    methods
        function obj = ELM(data)
            [obj.X, obj.T] = parseData(data);
        end
        
        function obj = addNeurons(obj, func, num)
            obj.Neurons = cat(1, obj.Neurons, {func, num});
        end
        
        function obj = train(obj)
            neuronsCount = sum(cell2mat(obj.Neurons(:, 2)));
            obj.W = random('Normal', 0, 1, size(obj.X, 2), neuronsCount);
            bLine = random('Normal', 0, 1, 1, neuronsCount);
            obj.B = repmat(bLine, size(obj.X), 1);
            obj.H = createH(obj);
            Hinv = pinv(obj.H);
            obj.Beta = Hinv * obj.T;
        end
        
        function obj = predict()
        end
        
        function [data, classes] = parseData(d)
            data = d(:, 1:end-1);
            classesVector = d(:, end);
            classes = zeros(size(data,1), max(classesVector));
            for i = 1:size(classesVector,1)
                classes(i, classesVector(i)) = 1;
            end
        end
        
        function retH = createH(obj)
            retH = [];
            neuronSum = 0;
            for neuron = obj.Neurons'
                Hi = func(obj.X * obj.W(neuronSum, neuronSum + neuron(2)) + obj.B);
                neuronSum = neuronSum + neuron(2);
                retH = cat(2, retH, Hi);
            end
        end
    end
end