classdef ELM < handle
    properties
        X
        T
        Neurons
        NeuronsCount
        W
        BLine
        H
        Beta
    end
    methods
        function obj = ELM(data)
            obj.parseData(data);
        end
        
        function obj = addNeurons(obj, funcStr, num)
            funcStr=strrep(funcStr, '^', '.^');
            funcStr=strrep(funcStr, '/', './');
            funcStr=strrep(funcStr, '*', '.*');
            command=sprintf('func=@(x)%s;', funcStr);
            eval(command);
            obj.Neurons = cat(1, obj.Neurons, {func, num});
        end
        
        function obj = train(obj)
            neuronsCount = sum(cell2mat(obj.Neurons(:, 2)));
            obj.W = random('Normal', 0, 1, size(obj.X, 2), neuronsCount);
            obj.BLine = random('Normal', 0, 1, 1, neuronsCount);
            obj.H = createH(obj, obj.X);
            Hinv = pinv(obj.H);
            obj.Beta = Hinv * obj.T;
        end
        
        function T = predict(obj, Data)
            H = createH(obj, Data);
            resultMatrix = H * obj.Beta;
            T = chooseResults(resultMatrix);
        end
        
        function res = chooseResults(m)
            tmp, res = max(m, [], 2);
        end
        
        function obj = parseData(obj, d)
            obj.X = d(:, 1:end-1);
            classesVector = d(:, end);
            obj.T = zeros(size(d, 1), max(classesVector));
            for i = 1:size(classesVector,1)
                obj.T(i, classesVector(i)) = 1;
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