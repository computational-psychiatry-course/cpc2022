% Samuel Taylor and Ryan Smith, 2021

% Three-arm bandit fitting script
% Fit a given simulated subject over all 20 blocks
% Should only be used with the output from `TAB_sim.m`
function FinalResults = TAB_sim_fit(sim_data)

    % Hard-coded trial structure (note this should be identical to the
    % matrix in `TAB_sim.m`).
   %     BlockProbs =   [0.6661	0.1447	0.7227;
%                     0.4557	0.4873	0.6372;
%                     0.2712	0.9723	0.7656;
%                     0.3623	0.3891	0.1726;
%                     0.0432	0.3888	0.7258;
%                     0.1321	0.671	0.7165;
%                     0.7151	0.5357	0.5668;
%                     0.6771	0.8909	0.1205;
%                     0.5586	0.5424	0.6091;
%                     0.2993	0.6921	0.1387;
%                     0.253	0.2638	0.8082;
%                     0.4937	0.8078	0.6343;
%                     0.5003	0.4056	0.3771;
%                     0.7886	0.6485	0.7971;
%                     0.7749	0.4079	0.428;
%                     0.697	0.251	0.3832;
%                     0.6982	0.5139	0.47;
%                     0.5797	0.5017	0.1139;
%                     0.4362	0.5075	0.2759;
%                     0.3672	0.0943	0.6427];

    BlockProbs =   [0.6661	0.1447	0.7227;
                    0.4557	0.4873	0.6372];
    
    
    %% 6.2 Invert model and try to recover original parameters:
    %==========================================================================

    %--------------------------------------------------------------------------
    % This is the model inversion part. Model inversion is based on variational
    % Bayes. The basic idea is to maximise (negative) variational free energy
    % wrt to the free parameters (here: alpha and cr). This means maximising
    % the likelihood of the data under these parameters (i.e., maximise
    % accuracy) and at the same time penalising for strong deviations from the
    % priors over the parameters (i.e., minimise complexity), which prevents
    % overfitting.
    % 
    % You can specify the prior mean and variance of each parameter at the
    % beginning of the TAB_spm_dcm_mdp script.
    %--------------------------------------------------------------------------
    
    TpB = 16;     % trials per block
%     NB  = 20;     % number of blocks
      NB  = 2;     % number of blocks
    N   = TpB*NB; % trials per block * number of blocks
    
    %--------------------------------------------------------------------------

    alpha    = 4;
    cr       = 4;
    beta     = 1;
    eta_win  = 1/2;
    eta_loss = 1/2;
    prior_a  = 1/4;

    MDP = TAB_gen_model(BlockProbs(1,:),beta,alpha,cr,eta_win,eta_loss,prior_a);

    MDP.BlockProbs = BlockProbs; % Block probabilities
    MDP.TpB        = TpB;        % trials per block
    MDP.NB         = NB;         % number of blocks
    MDP.prior_a    = prior_a;    % prior_a


    DCM.MDP    = MDP;                  % MDP model
    DCM.field  = {'alpha' 'cr' 'eta_win' 'eta_loss' 'prior_a'}; % Parameter field
    DCM.U      = {sim_data.observations};              % trial specification (stimuli)
    DCM.Y      = {sim_data.responses};              % responses (action)

    % Model inversion
    DCM        = TAB_inversion(DCM);

    %% 6.3 Check deviation of prior and posterior means & posterior covariance:
    %==========================================================================

    %--------------------------------------------------------------------------
    % re-transform values and compare prior with posterior estimates
    %--------------------------------------------------------------------------
    field = fieldnames(DCM.M.pE);
    for i = 1:length(field)
        if strcmp(field{i},'eta_win')
            prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
            posterior(i) = 1/(1+exp(-DCM.Ep.(field{i}))); 
        elseif strcmp(field{i},'eta_loss')
            prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
            posterior(i) = 1/(1+exp(-DCM.Ep.(field{i}))); 
        else
            prior(i) = exp(DCM.M.pE.(field{i}));
            posterior(i) = exp(DCM.Ep.(field{i}));
        end
    end
    
    all_MDPs = [];

    % Simulate beliefs using fitted values
    for block=1:NB
        sim_mdp = TAB_gen_model(BlockProbs(block, :), 1, posterior(1), posterior(2), posterior(3), posterior(4), posterior(5));
        
        % Deal for all TpB trials within a block
        MDPs(1:TpB) = deal(sim_mdp);
        
        for t=1:TpB
            MDPs(t).o = [1 sim_data.observations(t, block); 1 sim_data.responses(t, block)];
            MDPs(t).u = sim_data.responses(t, block);
        end

        % Run simulation routine
        MDPs  = spm_MDP_VB_X_eta2(MDPs);

        % Save block of MDPs to list of all MDPs
        all_MDPs = [all_MDPs; MDPs'];
        
        clear MDPs;
    end
    
    % Return input file name, prior, posterior, output DCM structure, and
    % list of MDPs across task using fitted posterior values
    FinalResults = [{'simulated'} prior posterior DCM all_MDPs];
end