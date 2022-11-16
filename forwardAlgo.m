function [p,alpha] = forwardAlgo(y,P,E,pi)

    % Set T and N to the lengths of the observation sequence and transition
    % matrix
    T = length(y);
    N = length(P(1,:));

    for i = 1:1:N 
        % Initialise of alpha at t=1 as log of initial probabilities + log
        % of emission probability from state i to first observation
        alpha(1,i) = log(pi(i))+log(E(i,y(1)));
    end

    % Iterating through t=2,..,T and i=x_t=1,...,N calculate the logarithm
    % of the sum over previous alpha (at time t-1) w.r.t j multiplied by the
    % transition probability from j to i. Then add the logarithm of the
    % emission probability from state i to y(t)
    for t = 2:1:T
        for i = 1:1:N
            alpha(t,i) = log(sum(exp(alpha(t-1,:))'.*P(:,i)))+log(E(i,y(t)));
        end
    end

    % Calculate the marginal probability by summing along the last row of
    % alpha and return this value as p
    p = sum(exp(alpha(T,:)));

end