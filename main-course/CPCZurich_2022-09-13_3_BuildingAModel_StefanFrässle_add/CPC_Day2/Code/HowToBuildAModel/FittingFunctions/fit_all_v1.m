function [BIC, iBEST, BEST] = fit_all_v1(a, r)

% fit all models and get the BIC
[~, ~, BIC(1)] = fit_M1random_v1(a, r);
[~, ~, BIC(2)] = fit_M2WSLS_v1(a, r);
[~, ~, BIC(3)] = fit_M3RescorlaWagner_v1(a, r);

% find the best model
[M, iBEST] = min(BIC);
BEST = BIC == M;
BEST = BEST / sum(BEST);

end