classdef ELM
    properties
        X
        T
        Neurons
        H
        Beta
    end
    methods
        function obj = ELM(data)
            [obj.X, obj.T] = parseData(data);
            obj.H = nan;
        end
        
        function [data, classes] = parseData(d)
            data = d(:, 1:end-1);
            classesVector = d(:, end);
            classes = zeros(size(data,1), max(classesVector));
            for i = 1:size(classesVector,1)
                classes(i, classesVector(i)) = 1;
            end
        end
        
        function obj = addNeurons(obj, func, num)
            % obj.Neurons = cat(1, 
        end
        
        function obj = train(obj)
            Hinv = pinv(obj.H);
            obj.Beta = Hinv * obj.T;
        end
        
        function predict()
            
        end
        
        function createH(data)
            W = random('Normal', 0, 1, size(obj.X, 2), num);
            biasLine = random('Normal', 0, 1, 1, num);
            B = repmat(biasLine, size(obj.Data), 1);
            Hi = func(obj.X*W + B);
            if isnan(obj.H)
                obj.H = Hi;
            else
                obj.H = [obj.H, Hi];
            end
        end
    end
end