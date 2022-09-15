function NegLL = lik_M3RescorlaWagner_v1(a, r, alpha, beta)

% initial values of the two slot machines
Q = [0.5 0.5];

% number of trials
T = length(a);

% loop over all trial
for t = 1:T
    
    % compute choice probabilities
    p = exp(beta*Q) / sum(exp(beta*Q));
    
    % compute choice probability for actual choice
    choiceProb(t) = p(a(t));
    
    % update values
    delta = r(t) - Q(a(t));
    Q(a(t)) = Q(a(t)) + alpha * delta;

end

% compute negative log-likelihood
NegLL = -sum(log(choiceProb));