% Auxiliary function for creating MDP model for active learning task
% Input:
% Rprob     = true reward prob in blocks
% beta      = hyper-prior on precision of policy selection (higher = less precise)
% alpha     = hyper-prior on precision of action selection  (higher = more precise)
% cr        = preference for win
% eta_win   = learning rate for wins
% eat_loss  = learning rate for losses
% prior_a   = insensitivity to information 
%
% Output:
% mdp model containing observation model, transition probs etc

% WAS: gen_mdp_Ryan_eta2
function mdp = TAB_gen_model(Rprob, beta, alpha, cr, eta_win, eta_loss, prior_a)

%% 12.1 Outcome probabilities: A
%==========================================================================

% Location and Reward, exteroceptive - no uncertainty about location, interooceptive - uncertainty about reward prob
%--------------------------------------------------------------------------
% States: start, left, middle, right (cols) --> outcomes: neutral, reward, no reward (rows)

probs = Rprob;

A{1} = [1 0            0            0            ; % reward neutral (starting position)
        0 probs(1)     probs(2)     probs(3)     ; % reward 
        0 (1-probs(1)) (1-probs(2)) (1-probs(3))]; % no reward

%States: start, left, middle, right (cols) --> outcomes: start, left, middle, right (rows)

A{2} = [1 0 0 0; % starting position
        0 1 0 0; % left choice 
        0 0 1 0; % middle choice
        0 0 0 1];% right choice
    
%% 12.2 Beliefs about outcome (likelihood) mapping
%==========================================================================

%--------------------------------------------------------------------------
% That's where learning comes in - start with uniform prior
%--------------------------------------------------------------------------
%prior_a = 1/4;

a{1} = [1 0       0       0       ; % reward neutral (starting position)
        0 prior_a prior_a prior_a ; % reward 
        0 prior_a prior_a prior_a]; % no reward
    
% States: start, left, middle, right (cols) --> outcomes: start, left, middle, right (rows)

a{2} = [1 0 0 0; % starting position
        0 1 0 0; % left choice 
        0 0 1 0; % middle choice
        0 0 0 1];% right choice    

%% 12.3 Controlled transitions: B{u}
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions of hidden states
% for each factor. Here, there are three actions taking the agent directly
% to each of the three locations.
%--------------------------------------------------------------------------
B{1}(:,:,1) = [1 1 1 1; 0 0 0 0;0 0 0 0;0 0 0 0];     % move to the starting point
B{1}(:,:,2) = [0 0 0 0; 1 1 1 1;0 0 0 0;0 0 0 0];     % choose left (and check for reward)
B{1}(:,:,3) = [0 0 0 0; 0 0 0 0;1 1 1 1;0 0 0 0];     % choose middle (and check for reward)
B{1}(:,:,4) = [0 0 0 0; 0 0 0 0;0 0 0 0;1 1 1 1];     % choose right (and check for reward)


%% 12.4 Priors: 
%==========================================================================

%--------------------------------------------------------------------------
% Finally, we have to specify the prior preferences in terms of log
% probabilities over outcomes. Here, the agent prefers high rewards over
% low rewards over no rewards
%--------------------------------------------------------------------------

C{1}  = [0 cr 0]'; % preference for: [staying at starting point | reward | no reward]
C{2}  = [0 0 0 0]';
%--------------------------------------------------------------------------
% Now specify prior beliefs about initial state
%--------------------------------------------------------------------------
D{1}  = [1 0 0 0]'; % prior over starting point - start, left, middle, right


%% 13.5 Allowable policies (of depth T).  These are sequences of actions
%==========================================================================

V     = [2 3 4]; % go left, go middle, go right

%% 14. Define MDP Structure
%==========================================================================
%==========================================================================

% Set up MDP

mdp.T = 2;                      % number of moves
mdp.V = V;                      % allowable policies (if deep)
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = D;                      % prior over initial states

mdp.a = a;                      % observation beliefs                     
 
mdp.Aname   = {'Win/Lose','Choice made'};
mdp.Bname   = {'Choice'};

mdp.beta    = beta;
mdp.alpha   = alpha;
mdp.eta_win = eta_win;
mdp.eta_loss = eta_loss;

mdp         = spm_MDP_check(mdp);


end