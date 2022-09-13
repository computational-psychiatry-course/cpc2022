%% Three-armed bandit task model building exercise

% First, you need to add SPM12, the DEM toolbox of SPM12 and the
% folder with the example scripts to your path in Matlab.

clear all
close all      % These commands clear the workspace and close any figures

rng('shuffle') % This sets the random number generator to produce a different 
               % random sequence each time, which leads to variability in 
               % repeated simulation results (you can alse set to 'default'
               % to produce the same random sequence each time)

% Simulation options after model building below:

% If Sim = 1, simulate single trial.

% If Sim = 2, simulate multiple trials.
             

Sim = 1;


%% 1. Set up model structure

% There will be two time points or 'epochs' within a trial: T
% =========================================================================

T = ??;

% Priors about initial states: D{f}
% =========================================================================

%--------------------------------------------------------------------------
% Specify prior probabilities about initial states in the generative 
% process (D) for each state factor f
% Note: By default, these will also be the priors for the generative model
%--------------------------------------------------------------------------

% How many state factors?

D{...} = ??; 
D{...} = ??; 

% State-outcome mappings in generative process: A{m}:
% =========================================================================

%--------------------------------------------------------------------------
% Specify the probabilities of outcomes given each state in the generative 
% process (A) for each outcome modality {m}
% This includes one matrix per outcome modality
% Note: By default, these will also be the beliefs in the generative model
% (a) if learning is enabled
%--------------------------------------------------------------------------

% Remember, the rows correspond to observations, the columns correspond
% to the first state factor, and the third dimension corresponds to the second
% state factor. Each column is a probability distribution that must sum to 1.

% How many A-matrices?

A{...}= ??;
A{...}= ??;

%--------------------------------------------------------------------------
% Specify prior beliefs about state-outcome mappings in the generative model: 
% a{m} for each outcome modality m. These are concentration parameters in
% Dirichlet distributions.
% Note: This will automatically simulate learning state-outcome mappings 
% if specified.
%--------------------------------------------------------------------------
           
% How many a-matrices?

a{...}= ??;
a{...}= ??;

% Controlled transitions: B{f}{:,:,u}
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions (B) between hidden states
% under each action u for each state factor f. 
%--------------------------------------------------------------------------

% Columns are states at time t. Rows are states at t+1. One 3rd B matrix dimension
% for each action (for each state factor)

% How many possible actions per state factor?

B{...} = ??;    
B{...} = ??; 
           
% Preferred outcomes: C 
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the 'prior preferences' C{m} for each outcome 
% modality m. Each column is transformed into a log-probability
% distribution, where higher "probability" indicates greater reward value. 
%--------------------------------------------------------------------------

% One matrix C per outcome modality m. Each row is an observation, and each
% column is a time point. Negative values indicate lower preference,
% positive values indicate a higher preference. Stronger preferences promote
% risky choices and reduced information-seeking.

% Specify preference strength: cr

cr = ??; % insert into C matrix

% How many C-matrices?

C{...}      = ??; 
C{...}      = ??;

% Allowable policies: U(:,p,f) or V(:,p,f). 
%==========================================================================

%--------------------------------------------------------------------------
% Each policy is a sequence of actions over time that the agent can 
% consider. Each policy p is a column, each state factor f is a 3rd
% dimension
%--------------------------------------------------------------------------

% Policies can be specified as 'shallow' (looking only one step
% ahead), as specified by U. Or policies can be specified as 'deep' 
% (planning actions all the way to the end of the trial), as specified by
% V. Both U and V must be specified for each state factor f as the 3rd
% matrix dimension. This will simply be all 1s if that state is not
% controllable.

% Here, each trial is only 1 action, so we can use U or V interchangably:
    
V     = ???; %

% Additional optional parameters. 
%==========================================================================

% Eta: learning rate (0-1) controlling the magnitude of concentration parameter
% updates after each trial (if learning is enabled).

    eta = ??; % By default we here set this to 0.5, but try changing its value  
               % to see how it affects model behavior

% Omega: forgetting rate (0-1) controlling the reduction in concentration parameter
% magnitudes after each trial (if learning is enabled). This controls the
% degree to which newer experience can 'over-write' what has been learned
% from older experiences. It is adaptive in environments where the true
% parameters in the generative process (priors, likelihoods, etc.) can
% change over time. A low value for omega can be seen as a prior that the
% world is volatile and that contingencies change over time.

    omega = ??;% By default we here set this to 1 (indicating no forgetting, 
               % but try changing its value to see how it affects model behavior. 
               % Values below 1 indicate greater rates of forgetting.
               
% Beta: Expected precision of expected free energy (G) over policies (a 
% positive value, with higher values indicating lower expected precision).
% Lower values increase the influence of habits (E) and otherwise make
% policy selection less deteriministic. For our example simulations we will
% simply set this to its default value of 1:

     beta = ??;% By default this is set to 1, but try increasing its value 
               % to lower precision and see how it affects model behavior

% Alpha: An 'inverse temperature' or 'action precision' parameter that 
% controls how much randomness there is when selecting actions (e.g., how 
% often the agent might choose not to take the hint, even if the model 
% assigned the highest probability to that action. This is a positive 
% number, where higher values indicate less randomness. Here we set this to 
% a high value:

    alpha = ??;  % Any positive number. 1 is very low, 32 is fairly high; 
                 % an extremely high value can be used to specify
                 % deterministic action (e.g., 512)

 
%% 2. Define MDP Structure
%==========================================================================
%==========================================================================

mdp.T = T;                    % Number of time steps
mdp.V = V;                    % policies (specifying U instead of V)
mdp.A = A;                    % state-outcome mapping
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred outcomes
mdp.D = D;                    % priors over initial states
    
mdp.eta = eta;                % learning rate
mdp.omega = omega;            % forgetting rate
mdp.alpha = alpha;            % action precision
mdp.beta = beta;              % expected precision of expected free energy over policies


% learning  parameters:
     mdp.a = a;  
     
% plotting

label.factor{1}   = 'Choice States';   label.name{1}    = {'Start','Left','Middle','Right'};
label.modality{1} = 'Outcomes';    label.outcome{1} = {'Start','Win','Lose'};
label.modality{2} = 'Observed Choices';  label.outcome{2} = {'Start','Left','Middle','Right'};
label.action{1} = {'Start','Left','Middle','Right'};
mdp.label = label;

%--------------------------------------------------------------------------
% Use a script to check if all matrix-dimensions are correct:
%--------------------------------------------------------------------------
mdp = spm_MDP_check(mdp);


if Sim ==1
%% 3. Single trial simulations
 
%--------------------------------------------------------------------------
% Now that the generative process and model have been generated, we can
% simulate a single trial using the spm_MDP_VB_X script. Here, we provide 
% a version specific to this tutorial - spm_MDP_VB_X_tutorial - that adds 
% forgetting rate (omega), which is not included in the current SPM version (as of 05/08/21).
%--------------------------------------------------------------------------

MDP = spm_MDP_VB_X_tutorial(mdp);

% We can then use standard plotting routines to visualize simulated neural 
% responses

spm_figure('GetWin','Figure 1'); clf    % display behavior
spm_MDP_VB_LFP(MDP); 

%  and to show posterior beliefs and behavior:

spm_figure('GetWin','Figure 2'); clf    % display behavior
spm_MDP_VB_trial(MDP); 

% Please see the main text for figure interpretations

elseif Sim == 2
%% 4. Multi-trial simulations

N = 16; % number of trials

MDP = mdp;

[MDP(1:N)] = deal(MDP);

MDP = spm_MDP_VB_X_tutorial(MDP);

% We can again visualize simulated neural responses

[sim_mdp, gen_data] = TAB_sim(alpha, cr, eta, eta, prior_a);
gtitle = sprintf('Simulated Data (for Alpha=%.2f; CR=%.2f; Eta Win=%.2f; Eta Loss=%.2f; Prior A=%.2f)', alpha, cr, eta, eta, prior_a);
TAB_plot(sim_mdp(1:16)', prior_a, gtitle);

end