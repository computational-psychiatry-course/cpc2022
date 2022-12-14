%% Simulate and/or fit subject data from three-armed bandit task
% Samuel Taylor and Ryan Smith, 2021

clear all
close all

% To fit a subject, specify the subject ID and directory of the subject in
% the variables below. The script will fit the subject, and if the plot
% flag is set, will plot the behavior of the subject overlayed on the
% beliefs produced by the model. 

% Before running, be sure to add spm12 and the DEM toolbox therein to your
% matlab path.

% SIM = false, FIT = true: fit a given subject (as specified by FIT_SUBJECT
% and INPUT_DIRECTORY). Will show a plot of action probabilities as
% determined by fitted parameters values, overlaid with the observations
% and responses of the true subject data.
% SIM = true, FIT = true: simulate behavior with the given parameter values
% (as specified by ALPHA, CR, ETA_WIN, ETA_LOSS, and PRIOR_A), and then fit
% to the simulated behavior. A good way to test parameter recoverability.
% Shows a plot of the original simulated behavior and accompanying actions
% probabilities, and will later show the action probabilities with the new
% fitted values as well.
% SIM = true, FIT = false: only simulate behavior. Shows a plot of
% simulated behavior and action probabilities.

SIM = true; % Generate simulated behavior (if false and FIT == true, will fit to subject file data instead)
FIT = true; % Fit example subject data 'TAB00' or fit simulated behavior (if SIM == true)

% Specify parameters (if simulating behavior)
ALPHA    = 10; % Specify ACTION PRECISION (ALPHA) parameter value
CR       = 3; % Specify REWARD SENSITIVITY (CR) parameter value
ETA_WIN  = 0.1; % Specify WIN LEARNING RATE (ETA_W) parameter value
ETA_LOSS = 0.8; % Specify LOSS LEARNING RATE (ETA_L) parameter value
PRIOR_A  = 0.25; % Specify INFORMATION INSENSITIVITY (ALPHA_0) parameter value

PLOT_FLAG = true; % Generate plot of behavior

FIT_SUBJECT = 'TAB00';   % Subject ID
INPUT_DIRECTORY = './';  % Where the subject file is located

if SIM
    [sim_mdp, gen_data] = TAB_sim(ALPHA, CR, ETA_WIN, ETA_LOSS, PRIOR_A);
    
    if PLOT_FLAG
        gtitle = sprintf('Simulated Data (for Alpha=%.2f; CR=%.2f; Eta Win=%.2f; Eta Loss=%.2f; Prior A=%.2f)', ALPHA, CR, ETA_WIN, ETA_LOSS, PRIOR_A);
        TAB_plot_fit_demo(sim_mdp(1:16)', PRIOR_A, gtitle);
        shg
    end
end

if FIT
    if SIM
        fit_results = TAB_sim_fit(gen_data);
        
        if PLOT_FLAG
            figure
            gtitle = sprintf('Fit Model (to Simulation Data Generated by Alpha=%.2f; CR=%.2f; Eta Win=%.2f; Eta Loss=%.2f; Prior A=%.2f)', ALPHA, CR, ETA_WIN, ETA_LOSS, PRIOR_A);
            TAB_plot_fit_demo(fit_results{5}(1:16)', fit_results{3}(5), gtitle);
        end
    else
        fit_results = TAB_fit(FIT_SUBJECT, INPUT_DIRECTORY);

        if PLOT_FLAG
            figure
            gtitle = sprintf('Fit Model (to %s Subject Data, Alpha=%.2f; CR=%.2f; Eta Win=%.2f; Eta Loss=%.2f; Prior A=%.2f)', FIT_SUBJECT, fit_results{3}(1), fit_results{3}(2), fit_results{3}(3), fit_results{3}(4), fit_results{3}(5));
            TAB_plot_fit_demo(fit_results{5}(1:16)', fit_results{3}(5), gtitle);
        end
    end

    fprintf('Fit: \n\tAlpha =\t%.3f\n\tCR =\t%.3f\n\tEta Win =\t%.3f\n\tEta Loss =\t%.3f\n\tPrior A =\t%.3f\n\tNegative Free Energy =\t%.3f\n', fit_results{3}(1), fit_results{3}(2), fit_results{3}(3), fit_results{3}(4), fit_results{3}(5),fit_results{1,4}.F);
end

return
%% Model Comparison

% Assume I have generated simulated data for 5 subjects from a model with 2 
% learning rates. Then I fit two models to that data: a 'one learning rate' 
% model (M1) and a 'two learning rate' model (M2).

%If the Fs for M1 fits are:
Fs_M1 = [-25.2 -24.4 -26.2 -28.6 -22.3]';

%and if the Fs for M2 fits are closer to 0:
Fs_M2 = Fs_M1 + 2;

% then we can do Bayesian Model Comparison using the spm_BMS.m function:
[alpha,exp_r,xp,pxp,bor] = spm_BMS([Fs_M1 Fs_M2]);

disp(' ');
disp(' ');
disp('Protected exceedance probability (pxp):');
disp(pxp);
disp(' ');