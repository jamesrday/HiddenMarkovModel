function beta = backwardAlgo(y,P,E)

    % Set T and N to the lengths of the observation sequence and transition
    % matrix
    T = length(y);
    N = length(P(1,:));

    for i = 1:1:N     
       % Initialise beta at T to be 1 (logarithm is 0)
       beta(T,i) = 0;
    end
    
    % Iterating over t=T-1,...,1 and i=x_t=1,...,N calculate the logarithm 
    % of the sum of beta at t+1 * the probability of emission from j to 
    % y(t+1) * the probability of transitioning from i to j
    for t = (T-1):-1:1
        for i = 1:1:N
            beta(t,i) = log(sum(P(i,:).*E(:,y(t+1))'.*exp(beta(t+1,:))));
        end
    end
end