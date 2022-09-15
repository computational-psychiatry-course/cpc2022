function [Xfit, LL, BIC] = fit_M1random_v1(a, r)

% define the likelihood function of random model
obFunc = @(x) lik_M1random_v1(a, r, x);

% define options for the fmin search
options = optimoptions('fmincon','Display','off');

% set initial condition of algorithm and lower and upper bound
X0 = rand;
LB = 0;
UB = 1;

% perform minimization
[Xfit, NegLL] = fmincon(obFunc, X0, [], [], [], [], LB, UB, [], options);

% get log-likelihood and BIC
LL = -NegLL;
BIC = length(X0) * log(length(a)) + 2*NegLL;

end