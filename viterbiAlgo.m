function [argMaxDelta,delta] = viterbiAlgo(y,P,E,pi)

    % Set T and N to the lengths of the observation sequence and transition
    % matrix
    T = length(y);
    N = length(P(1,:));
    
    for i = 1:1:N 
        % Initialise first row of delta as log of the initial probability +
        % log of the emission probability from i to the first observation
        delta(1,i) = log(pi(i))+log(E(i,y(1)));
    end

    % Iterating through t=2,...,T and i=1,...,N replace sum with max in the
    % forward algorithm to calculate delta
    for t = 2:1:T
        for i = 1:1:N
            delta(t,i) = max(delta(t-1,:)+log(P(:,i)')+log(E(i,y(t))));
        end
        % Store the argmax of delta at row t in argMaxDelta(t) which will
        % be returned along with delta. Note that for q2 path correcting is
        % not required. For the path corrected version to be used in Q4 see
        % viterbiAlgoPathCorrected.m
        [~,argMaxDelta(t-1)]=max(delta(t-1,:));
    end

    % Calculate final value of most likely state sequence
    [~,argMaxDelta(T)]=max(delta(T,:));
    
end