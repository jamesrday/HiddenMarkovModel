function [argMaxDelta,delta] = viterbiAlgoPathCorrected(y,P,E,pi)

    % Set T and N to the lengths of the observation sequence and transition
    % matrix
    T = length(y);
    N = length(P(1,:));

    % Allocate row vector of zeroes of length T
    argMaxDelta = zeros(1,T);
    
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
    end

    % Check for valid path
    for count=1:1:(T-1)
        % Set argMaxDelta at t=count to usual value (most likely but 
        % possibly invalid path)
        [~, argMaxDelta(count)] = max(delta(count,:));

        % For all most likely hidden states excel ell_i check whether the
        % bottom right neighbour in the trellis of each delta is delta=-inf.
        % If so, correct path to second-most likely path etc. by replacing
        % delta at row count and current state with -inf. Note that
        % log(0)=-inf. If there is no valid path the program will never
        % halt.
        if argMaxDelta(count)~=N
            while delta(count+1,argMaxDelta(count)+1)==-inf && argMaxDelta(count)~=4
                fprintf('Corrected to valid path at t=%g\n', count);
                delta(count,argMaxDelta(count))=-inf;
                [~, argMaxDelta(count)] = max(delta(count,:));
            end
        end
    end

    % Calculate final value of most likely state sequence
    [~,argMaxDelta(T)]=max(delta(T,:));
    
end