function [Xfit, LL, BIC] = fit_M3RescorlaWagner_v1(a, r)

% define the likelihood function of Rescorla-Wagner model
obFunc = @(x) lik_M3RescorlaWagner_v1(a, r, x(1), x(2));

% define options for the fmin search
options = optimoptions('fmincon','Display','off');

% set initial condition of algorithm and lower and upper bound
X0 = [rand exprnd(1)];
LB = [0 0];
UB = [1 inf];

% perform minimization
[Xfit, NegLL] = fmincon(obFunc, X0, [], [], [], [], LB, UB, [], options);

% get log-likelihood and BIC
LL = -NegLL;
BIC = length(X0) * log(length(a)) + 2*NegLL;

end