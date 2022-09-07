%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- Hierarchical Model Tutorial --%
%----------- Exercises -----------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CPC 2022 Active Inference Tutorial:
% Hiearchical model excersises based on the supplementary Code for: A Step-by-Step 
% Tutorial on Active Inference Modelling and its Application to Empirical Data

% Ryan Smith, Rowan Hodson, and Christopher Whyte

% Step by step tutorial for building hierarchical POMDPs using the active
% inference framework. Here we simulate the now classic "Local Global" auditory
% mismatch paradigm. This will reproduce results similar to figs. 14-16 in
% the paper. 

clear
close all

%% Level 1: Perception of individual stimuli
%==========================================================================

% prior beliefs about initial states
%--------------------------------------------------------------------------

D{1} = [1 1]';% stimulus tone {high, low}

d = D;

% Here we seperate the generative process (the capital D)
% from the generative model (the lower case d) allowing learning to occur
% (i.e. to acccumulate concentration paramaters) in the generative model, 
% independent of the generative process.

% probabilistic (likelihood) mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------- Exercise 1 --------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Based on what you learnt in the first level model exercises:
% a) write down the liklihood matrix A{1} between hidden tone states (high, low)
% and tone observations (high, low) as an identity matrix.

%---- Write code for exercise 1a here

% outcome modality 1: stimulus tone
A{1}= [1 0;  %high tone
       0 1]; %low tone
   

% b) seperate the generative model a{1} from the generative process A{1}, reduce the
% precision of the mapping (precision = 2) by passing the a{1} matrix through a softmax 
% function (hint use the function spm_softmax()), and multiple the generative process by 100 
% to prevent learning. 

%---- Write code for exercise 1b here

% seperate generative model from generative process
a = A;

% reduce precision
pr1 = 2; % precision (inverse termperature) parameter (lower = less precise)
a{1} = spm_softmax(pr1*log(A{1}+exp(-4)));

a = a{1}*100;


% Transitions between states: B
%--------------------------------------------------------------------------

B{1}= [1 0; %high tone
       0 1];%low tone

% MDP Structure
%--------------------------------------------------------------------------
mdp_1.T = 1;                      % number of updates
mdp_1.A = A;                      % likelihood mapping
mdp_1.B = B;                      % transition probabilities
mdp_1.D = D;                      % prior over initial states
mdp_1.d = d;
mdp_1.a = a;
mdp_1.erp = 1;

mdp_1.Aname = {'Stimulus'};
mdp_1.Bname = {'Stimulus'};

clear a d A B D

MDP_1 = spm_MDP_check(mdp_1);

clear mdp_1

%% Level 2: Slower-timescale representations of perceived stimulus sequences
%==========================================================================

% prior beliefs about initial states in generative process (D) and
% generative model (d) in terms of counts (i.e., concentration parameters)
%--------------------------------------------------------------------------
D2{1} = [1 1 1 1]'; % Sequence type: {high, low, high-low, low-high}
D2{2} = [1 0 0 0 0 0]'; % time-in-trial
D2{3} = [1 0 0]'; % Report: {null, same, different} 

d2 = D2;
d2{2} = d2{2}*100;
d2{3} = d2{3}*100;

% Again, we here seperate the generative model from the generative process,
% and multiply d2{2} and d2{3} by 100 to prevent learning in the model's
% representation of task phase (time in trial) and report state probabilities.

% probabilistic (likelihood) mapping from hidden states to outcomes: A
% outcomes: A{1} stim (2), A{2} Report Feedback (3)
%--------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------- Exercise 2 --------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Write down the second level A matrix A2{1}(liklihood) to map between the second
% level hidden states (high, low, high-low, low-high), which encode beliefs about 
% sequences of hidden states at the first level, and first level hidden states (low, high) 
% which now play the role of observations. The second level hidden states high and low 
% should map to high and low observations for all 6 time steps. high-low and low-high hidden
% states should map to high and low observations for all times except the fouth time-step 
% which should be an odd ball with the high-low hidden state generating a low observation, 
% and low-high hidden state generating a high observation. 

% HINT 1: the easiest way to do this is with two for loops. In the first for 
% loop specify the behaviour for all trials except time-in-trial = 4, and in the second 
% specify the observations just for time-in-trial = 4. 

% Hint 2: remember that we are working with factorised distributions where each dimension 
% in the A matrix (A tensor) plays the role as hidden states play in a conditional distribution 
% (i.e. A(:,:,i,j) = p(o = rows|sequence type = columns,time-in-trial = i,report = j)

%---- Write code for exercise 2 here

%--- Stimulus
for i = 1:6
    for j = 1:3
        A2{1}(:,:,i,j) = [1 0 1 0; %high
                          0 1 0 1];%low
    end 
end

% oddball at fourth timestep for high-low and low-high hidden states. 
for i = 4
    for j = 1:3
        A2{1}(:,:,i,j) = [1 0 0 1; %high
                          0 1 1 0];%low
    end
end

%--- Report
for i = 1:6
    for j = 1:3
        A2{2}(:,:,i,j) = [1 1 1 1; %null
                          0 0 0 0; %incorrect
                          0 0 0 0];%correct                    
    end
end

% report "same"
for i = 6
    for j = 2
        A2{2}(:,:,i,j) = [0 0 0 0; %null
                          0 0 1 1; %incorrect
                          1 1 0 0];%correct
    end
end

% report "different"
for i = 6
    for j = 3
        A2{2}(:,:,i,j) = [0 0 0 0; %null
                          1 1 0 0; %incorrect
                          0 0 1 1];%correct
    end
end

a2 = A2; % likelihood (concentration parameters) for generative model

% reduce precision
pr2 = 2; % precision (inverse termperature) parameter (lower = less precise)
a2{1} = spm_softmax(pr2*log(A2{1}+exp(-4)));

a2{1} = a2{1}*100;
a2{2} = a2{2}*100;

% Transition probabilities: B
%--------------------------------------------------------------------------

% Precision of sequence mapping
B2{1} = eye(4,4); % maximally precise identity matrix (i.e., the true 
                  % sequence is stable within a trial)
                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------- Exercise 3 --------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Write down the time-in-trial transition matrix so that each state
% deterministically transitions to the next 1 -> 2 -> 3 ... 6, and state
% six stays in state six. 

%---- Write code for exercise 3 here

B2{2} = [0 0 0 0 0 0;
         1 0 0 0 0 0;
         0 1 0 0 0 0;
         0 0 1 0 0 0;
         0 0 0 1 0 0;
         0 0 0 0 1 1]; % Deterministically transition through trial sequence
 
     
% Report
B2{3}(:,:,1) = [1 1 1;
                0 0 0;
                0 0 0]; % Pre-report    
B2{3}(:,:,2) = [0 0 0;
                1 1 1;
                0 0 0]; % Report "same"   
B2{3}(:,:,3) = [0 0 0;
                0 0 0;
                1 1 1]; % Report "different"
           
% Policies
%--------------------------------------------------------------------------

 T = 6;  % number of timesteps
 Nf = 3; % number of factors
 Pi = 2; % number of policies
 V2 = ones(T-1,Pi,Nf);

% Report: "same" (left column) or "different" (right column)
 V2(:,:,3) = [1 1; 
              1 1;
              1 1;
              1 1;
              2 3];

% C matrices (outcome modality by timestep)
%--------------------------------------------------------------------------
C2{1} = zeros(2,T);

% report
C2{2} = [0 0 0 0 0 0;  % no feedback yet
         0 0 0 0 0 -1; % preference not to be incorrect at last timestep
         0 0 0 0 0 1]; % preference for being correct at last timestep

% MDP Structure
%--------------------------------------------------------------------------

mdp.MDP = MDP_1; % lower level model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------- Exercise 4 --------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Here we need to specify the linking matrix (mdp.link = ...) that tells the 
% model which first level hidden states play the role of observations at the 
% second level, and which second level hidden states play the role of the D 
% vector at the first level. The rows of the link matrix encode the first level
% hidden state factors, and the columns encode the second level observation
% modalites. Here we have one hidden state factor at the first level and two
% second level outcome modalities. The aim of this exercise is to write
% down a linking matrix that maps the only hidden state factor at the first level  
% (D{1}) to the first outcome modalitysecond level (A2{1}) whilst leaving 
% the second outcome modality (A{2}) independent of the first level.

% HINT: in a model with two first level hidden state factors and three
% second level observation modalities the link function needs to be a
% 2x3 matrix. To link the second outcome modality at the first level with
% the first outcome modality at the second level we would write:
% mpd.link = [0 0 0;
%             1 0 0];        

%---- Write code for exercise 4 here

mdp.link = [1 0]; % identifies lower level state factors (rows) with higher  
                  % level observation modalities (columns). Here this means the
                  % first observation at the higher level corresponds to
                  % the first state factor at the lower level.

mdp.T = T;                      % number of time points
mdp.A = A2;                     % likelihood mapping for generative process
mdp.a2 = a2;                    % likelihood mapping for generative model
mdp.B = B2;                     % transition probabilities
mdp.C = C2;                     % preferred outcomes
mdp.D = D2;                     % priors over initial states for generative process
mdp.d = d2;                     % priors over initial states for generative model
mdp.V = V2;                     % policies
mdp.erp = 1;                    % reset/decay paramater

mdp.Aname = {'Stimulus', 'Report Feedback'};
mdp.Bname = {'Sequence', 'Time in trial', 'Report'};


% level one labels 
label.factor{1}   = 'Stimulus';   label.name{1}    = {'High','Low'};
label.modality{1} = 'Stimulus';   label.outcome{1} = {'High','Low'};
mdp.MDP.label = label;

label.factor{1}   = 'Sequence type';   label.name{1}    = {'High','Low','High-low','Low-high'};
label.factor{2}   = 'Time in trial';    label.name{2}    = {'T1', 'T2', 'T3', 'T4', 'T5', 'T6'};
label.factor{3}   = 'Report';    label.name{3}    = {'Null', 'Same', 'Different'};
label.modality{1} = 'Tone';    label.outcome{1} = {'High', 'Low'};
label.modality{2} = 'Feedback';  label.outcome{2} = {'Null','Incorrect','Correct'};
label.action{3} = {'Null','Same','Different'};
mdp.label = label;

mdp = spm_MDP_check(mdp);
MDP = spm_MDP_VB_X_tutorial(mdp);

%Plot trial
spm_figure('GetWin','trial'); clf
spm_MDP_VB_trial(MDP);

%% Simulate all conditions

% Here we specify the number of trials N and use a deal function (which copies 
% the input to N outputs) to create 10 identical mdp structures. We can
% then pass this to the spm_MDP_VB_X_tutorial() script, which sequentially updates
% the concentration paramaters aquired on each trial and passes them to the
% mdp structure for the next trial (allowing learning to occur).

N = 10; %number of trials

% Local deviation - global standard
mdp.s = 3; % first nine trials are high-low 
MDP_condition1(1:N) = deal(mdp);
MDP_condition1(10).s = 3; % tenth trial is also high-low 
MDP_LDGS = spm_MDP_VB_X_tutorial(MDP_condition1);

% Local standard - global deviation
mdp.s = 3; % first nine trials are high-low
MDP_condition2(1:N) = deal(mdp);
MDP_condition2(10).s = 1; % tenth trial is a high trial 
MDP_LSGD = spm_MDP_VB_X_tutorial(MDP_condition2);

%% Plot ERPs using standard routines for each of the four conditions

% These are slightly modified versions of the standard plotting scripts
% given in the SPM software.

spm_figure('GetWin','ERP T1 - Local deviation - global standard'); clf
spm_MDP_VB_ERP_tutorial(MDP_LDGS(1));
spm_figure('GetWin','Trial T1 - Local deviation - global standard'); clf
spm_MDP_VB_trial(MDP_LDGS(1));
spm_figure('GetWin','ERP T10 - Local deviation - global standard'); clf
spm_MDP_VB_ERP_tutorial(MDP_LDGS(10));
spm_figure('GetWin','Trial T10 - Local deviation - global standard'); clf
spm_MDP_VB_trial(MDP_LDGS(10));

spm_figure('GetWin','ERP T1 - Local standard - global deviation'); clf
spm_MDP_VB_ERP_tutorial(MDP_LSGD(1));
spm_figure('GetWin','Trial T1 - Local standard - global deviation'); clf
spm_MDP_VB_trial(MDP_LSGD(1));
spm_figure('GetWin','ERP T10 - Local standard  - global deviation'); clf
spm_MDP_VB_ERP_tutorial(MDP_LSGD(10));
spm_figure('GetWin','Trial T10 - Local standard - global deviation'); clf
spm_MDP_VB_trial(MDP_LSGD(10));

%% custom ERP plots

% The ERP plotting routines give three outputs: 
% [level 2 ERPs, level 1 ERPs, indices]
% There are 32 time indices per time step/epoch of gradient decent. Here   
% there are 6 timesteps so there are 32x6 = 192 individual time indexes.
% The level 1 and 2 ERPs are the first derivitives at each time index.

[u1_1,v1_1,ind] = spm_MDP_VB_ERP_tutorial(MDP_LDGS(1),1);  
[u1_10,v1_10] = spm_MDP_VB_ERP_tutorial(MDP_LDGS(10),1); 

[u2_1,v2_1] = spm_MDP_VB_ERP_tutorial(MDP_LSGD(1),1);  
[u2_10,v2_10] = spm_MDP_VB_ERP_tutorial(MDP_LSGD(10),1); 

% The indexes below are arbitarily chosen to best represent the ERPs at the
% 4th time step, which starts at 96ms and ends at 128ms. To do this for 
% yourself we recommend just plotting the ERPs and selecting the appropiate
% time window. For example, the 1st level ERPs start at the begining of 
% the epoch whereas the 2nd ERPs appear towards the end of the epoch. So to
% include baseline periods in the plot you will likley have to select 
% slightly different time windows for each level as we have done here.

% index into 2nd level
index = (96:140); 
u1_1  = u1_1(index,:); % level 2
u1_10 = u1_10(index,:);

u2_1  = u2_1(index,:);% level 2
u2_10  = u2_10(index,:);

% index into ist level
index = (70:120); 
v1_1  = v1_1(index,:);% level 1
v1_10  = v1_10(index,:);
v2_1  = v2_1(index,:);% level 1
v2_10  = v2_10(index,:);

time_low = (1:length(v1_1)); 
time_high = (1:length(u1_1)); 

%--- Lets make the plots! 

% low level plot
limits = [20 45 -.5 1.2];

figure(10)
hold on
plot(time_low,sum(v2_10,2),'b','LineWidth',4) % local standard
plot(time_low,sum(v1_10,2),'r','LineWidth',4) % local deviation
axis(limits)
set(gca,'FontSize',10)
title('Mismatch negativity')
legend('Local standard', 'Local deviation')

% high level plot
limits = [1 45 -.5 .5];

figure(11)
hold on
plot(time_high,sum(u1_10,2),'b','LineWidth',4) % Global standard
plot(time_high,sum(u2_10,2),'r','LineWidth',4) % Global deviation
axis(limits)
set(gca,'FontSize',10)
title('P300')
legend('Global standard', 'Global deviation')

% MMN (standard - mismatch)
limits = [20 45 -1.2 .5];

figure(12)
hold on
plot(time_low,sum(v2_10-v1_10,2),'k','LineWidth',4) 
axis(limits)
set(gca,'FontSize',10)
title('Mismatch negativity: local standard - local deviation')

% P300 (standard - mismatch)
limits = [1 45 -.5 .5];

figure(13)
hold on
plot(time_high,sum(u1_10-u2_10,2),'k','LineWidth',4) 
axis(limits)
set(gca,'FontSize',10)
title('P300: Global standard - Global deviation')
