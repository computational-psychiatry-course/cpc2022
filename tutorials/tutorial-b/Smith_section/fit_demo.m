clear all
close all

% Specify parameters (if simulating behavior)
ALPHA    = 4; % Specify ACTION PRECISION (ALPHA) parameter value
CR       = 4; % Specify REWARD SENSITIVITY (CR) parameter value
ETA_WIN  = 0.5; % Specify WIN LEARNING RATE (ETA_W) parameter value
ETA_LOSS = 0.5; % Specify LOSS LEARNING RATE (ETA_L) parameter value
PRIOR_A  = .25; % Specify INFORMATION INSENSITIVITY (ALPHA_0) parameter value

% Task configuration
BlockProbs =   [0.2993	0.6921	0.1387];
TpB = 16; % Trials per Block

% Create generative model
mdp = TAB_gen_model(BlockProbs(:), 1, ALPHA, CR, ETA_WIN, ETA_LOSS, PRIOR_A);
        
% Deal for all NB*TpB trials
MDP(1:TpB) = deal(mdp);

% Hard-coded actions
actions = [2 3 4 3 3 3 3 3 2 4 3 3 3 3 3 4];
obs     = [3 2 3 2 2 2 2 3 3 3 2 2 2 2 3 3];

for t=1:TpB
    MDP(t).u = actions(t);
    MDP(t).o = [1 obs(t); 1 actions(t)];
end

% Run simulation routine
MDP  = spm_MDP_VB_X_eta2(MDP);

for t=1:TpB
    MDP(t).u = actions(t);
end

figure
gtitle = sprintf('Simulated Data (for Alpha=%.2f; CR=%.2f; Eta Win=%.2f; Eta Loss=%.2f; Prior A=%.2f)', ALPHA, CR, ETA_WIN, ETA_LOSS, PRIOR_A);
TAB_plot_fit_demo(MDP(1:16), PRIOR_A, gtitle);