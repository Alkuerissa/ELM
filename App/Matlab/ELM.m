classdef ELM < handle
    properties
        X
        T
        testX
        testT
        Neurons
        NeuronsCount
        W
        BLine
        H
        Beta
    end
    methods
        function obj = ELM(data, trainingPercentage)
            if nargin < 2
               trainingPercentage = 80;
            end
            if trainingPercentage > 95
                trainingPercentage = 95;
            end
            if trainingPercentage < 10
                trainingPercentage = 10;
            end
            trainingNum = int32(trainingPercentage * size(data, 1) / 100);
            [obj.X, obj.T] = obj.parseData(data(1:trainingNum, :));
            obj.X = obj.normalize(obj.X);
            d = data(trainingNum+1:end, :);
            obj.testX = d(:, 1:end-1);
            obj.testX = obj.normalize(obj.testX);
            obj.testT = d(:, end);
        end
        
        function obj = addNeurons(obj, funcStr, num)
            funcStr=strrep(funcStr, '^', '.^');
            funcStr=strrep(funcStr, '/', './');
            funcStr=strrep(funcStr, '*', '.*');
            command=sprintf('func=@(x)%s;', funcStr);
            eval(command);
            obj.Neurons = cat(1, obj.Neurons, {func, num});
        end
        
        function normalized = normalize(~, data)
            minimum = min(data);
            maximum = max(data);
            diff = (maximum - minimum);
            for i = 1:size(diff, 2)
                if diff(i) == 0
                    diff(i) = maximum(i);
                end
                if diff(i) == 0
                    diff(i) = 1;
                end
            end
            normalized = (data - minimum)./diff;
        end
        
        function obj = train(obj)
            neuronsCount = sum(cell2mat(obj.Neurons(:, 2)));
            obj.W = random('Normal', 0, 1, size(obj.X, 2), neuronsCount);
            obj.BLine = random('Normal', 0, 1, 1, neuronsCount);
            obj.H = createH(obj, obj.X);
            Hinv = pinv(obj.H);
            obj.Beta = Hinv * obj.T;
        end
        
        function T = predict(obj)
            H = createH(obj, obj.testX);
            resultMatrix = H * obj.Beta;
            T = obj.chooseResults(resultMatrix);
        end
        
        function res = exactCompare(~, actualT, predictedT)
            res = double(sum(actualT == predictedT)) / size(actualT, 1);
        end
        
        function res = meanDistanceCompare(~, actualT, predictedT)
            res = mean(abs(actualT - predictedT));
        end
        
        function res = chooseResults(~, m)
            [~, res] = max(m, [], 2);
        end
        
        function [X, T] = parseData(~, d)
            X = d(:, 1:end-1);
            classesVector = d(:, end);
            T = zeros(size(d, 1), max(classesVector));
            for i = 1:size(classesVector,1)
                T(i, classesVector(i)) = 1;
            end
        end
        
        function retH = createH(obj, X)
            retH = [];
            neuronSum = 1;
            B = repmat(obj.BLine, size(X, 1), 1);
            for neuron = obj.Neurons'
                Hi = neuron{1}(X * obj.W(:, neuronSum:neuronSum + neuron{2} - 1) + B);
                neuronSum = neuronSum + neuron{2};
                retH = cat(2, retH, Hi);
            end
        end
    end
end