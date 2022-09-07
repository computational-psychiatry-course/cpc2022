% Samuel Taylor and Ryan Smith, 2021

% Model inversion script
function [DCM] = TAB_inversion(DCM)

% MDP inversion using Variational Bayes
% FORMAT [DCM] = spm_dcm_mdp(DCM)

% If simulating - comment out section on line 196
% If not simulating - specify subject data file in this section 

%
% Expects:
%--------------------------------------------------------------------------
% DCM.MDP   % MDP structure specifying a generative model
% DCM.field % parameter (field) names to optimise
% DCM.U     % cell array of outcomes (stimuli)
% DCM.Y     % cell array of responses (action)
%
% Returns:
%--------------------------------------------------------------------------
% DCM.M     % generative model (DCM)
% DCM.Ep    % Conditional means (structure)
% DCM.Cp    % Conditional covariances
% DCM.F     % (negative) Free-energy bound on log evidence
% 
% This routine inverts (cell arrays of) trials specified in terms of the
% stimuli or outcomes and subsequent choices or responses. It first
% computes the prior expectations (and covariances) of the free parameters
% specified by DCM.field. These parameters are log scaling parameters that
% are applied to the fields of DCM.MDP. 
%
% If there is no learning implicit in multi-trial games, only unique trials
% (as specified by the stimuli), are used to generate (subjective)
% posteriors over choice or action. Otherwise, all trials are used in the
% order specified. The ensuing posterior probabilities over choices are
% used with the specified choices or actions to evaluate their log
% probability. This is used to optimise the MDP (hyper) parameters in
% DCM.field using variational Laplace (with numerical evaluation of the
% curvature).
%
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_mdp.m 7120 2017-06-20 11:30:30Z spm $

% OPTIONS
%--------------------------------------------------------------------------
ALL = false;

% prior expectations and covariance
%--------------------------------------------------------------------------
prior_variance = 2^-2;

for i = 1:length(DCM.field)
    field = DCM.field{i};
    try
        param = DCM.MDP.(field);
        param = double(~~param);
    catch
        param = 1;
    end
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        if strcmp(field,'alpha')
            pE.(field) = log(4);                % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'prior_a')
            pE.(field) = log(1/4);              % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'cr')
            pE.(field) = log(4);                % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'eta_win')
            pE.(field) = log(0.5/(1-0.5));      % in logit-space - bounded between 0 and 1!
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'eta_loss')
            pE.(field) = log(0.5/(1-0.5));      % in logit-space - bounded between 0 and 1!
            pC{i,i}    = prior_variance;
        else
            pE.(field) = 0;      
            pC{i,i}    = prior_variance;
        end
    end
end

pC      = spm_cat(pC);

% model specification
%--------------------------------------------------------------------------
M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
M.pE    = pE;                            % prior means (parameters)
M.pC    = pC;                            % prior variance (parameters)
M.mdp   = DCM.MDP;                       % MDP structure

% Variational Laplace
%--------------------------------------------------------------------------
[Ep,Cp,F] = spm_nlsi_Newton(M,DCM.U,DCM.Y);

% Store posterior densities and log evidnce (free energy)
%--------------------------------------------------------------------------
DCM.M   = M;
DCM.Ep  = Ep;
DCM.Cp  = Cp;
DCM.F   = F;


return

function L = spm_mdp_L(P,M,U,Y)
% log-likelihood function
% FORMAT L = spm_mdp_L(P,M,U,Y)
% P    - parameter structure
% M    - generative model
% U    - inputs
% Y    - observed repsonses
%__________________________________________________________________________

% multiply parameters in MDP
%--------------------------------------------------------------------------
mdp   = M.mdp;
field = fieldnames(M.pE);
for i = 1:length(field)
    if strcmp(field{i},'alpha')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'prior_a')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'cr')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'eta_win')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    elseif strcmp(field{i},'eta_loss')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    else
        mdp.(field{i}) = exp(P.(field{i}));
    end
end

U_block = U{:};
U_block = reshape(U_block,mdp.TpB,mdp.NB);

Y_block = Y{:};
Y_block = reshape(Y_block,mdp.TpB,mdp.NB);

L = 0;

% Each block is separate -- effectively resetting beliefs at the start of
% each block. 
for idx_block = 1:mdp.NB

    mdp_block = TAB_gen_model(mdp.BlockProbs(idx_block,:),mdp.beta,mdp.alpha,mdp.cr,mdp.eta_win,mdp.eta_loss,mdp.prior_a);

    [MDP(1:mdp.TpB)]   = deal(mdp_block);
    for idx_trial = 1:size(MDP,2)
        MDP(idx_trial).o = [1 U_block(idx_trial,idx_block); 1 Y_block(idx_trial,idx_block)];
        MDP(idx_trial).u = Y_block(idx_trial,idx_block);
    end

    % solve MDP and accumulate log-likelihood
    %--------------------------------------------------------------------------
    MDP  = spm_MDP_VB_X_eta2(MDP);

    for i = 1:numel(Y)
        for j = 1:mdp.TpB
            L = L + log(MDP(j).P(Y_block(j,idx_block)) + eps);
        end
    end

    clear('MDP')

end

fprintf('LL: %f \n',L)


