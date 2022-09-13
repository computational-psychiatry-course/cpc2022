% Samuel Taylor and Ryan Smith, 2021

% Plots the action probabilities, observations, and responses for the TAB 
% task
function [] = TAB_plot_fit_demo(MDP, prior_a, ggtitle)

% auxiliary plotting routine for spm_MDP_VB - multiple trials, curiosity
% paper
%
% MDP.P(M,T)      - probability of emitting action 1,...,M at time 1,...,T
% MDP.Q(N,T)      - an array of conditional (posterior) expectations over
%                   N hidden states and time 1,...,T
% MDP.X           - and Bayesian model averages over policies
% MDP.R           - conditional expectations over policies
% MDP.O(O,T)      - a sparse matrix encoding outcomes at time 1,...,T
% MDP.S(N,T)      - a sparse matrix encoding states at time 1,...,T
% MDP.U(M,T)      - a sparse matrix encoding action at time 1,...,T
% MDP.W(1,T)      - posterior expectations of precision
%
% MDP.un  = un    - simulated neuronal encoding of hidden states
% MDP.xn  = Xn    - simulated neuronal encoding of policies
% MDP.wn  = wn    - simulated neuronal encoding of precision
% MDP.da  = dn    - simulated dopamine responses (deconvolved)
% MDP.rt  = rt    - simulated dopamine responses (deconvolved)
%
% please see spm_MDP_VB
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% Philipp Schwartenbeck

% graphics
%==========================================================================
% col   = {'.b','.y','.g','.r','.c','.k'};
col   = {[0, 0.4470, 0.7410], ...       % blue
         [0.4660, 0.6740, 0.1880], ...  % green
         [0.9350, 0.1780, 0.2840], ...  % red
         [0.4940, 0.1840, 0.5560], ...  % purple
         [0.3010, 0.7450, 0.9330], ...  % cyan
         [0, 0, 0]};                    % black
cols  = [0:1/32:1; 0:1/32:1; 0:1/32:1]';

n_trials   = size(MDP,2);               % number of trials
n_timestep = size(MDP(1).V,1) + 1;      % number of time steps per trial

% MarkerSize = [24 24 24 24 24 24];
MarkerSize = 16;

for i = 1:n_trials
    
    % assemble performance
    %----------------------------------------------------------------------
    p(i)  = 0;
    
    for g = 1:numel(MDP(1).A)
        
        U = spm_softmax(MDP(i).C{g});
        
        for t = 1:n_timestep
            p(i) = p(i) + log(U(MDP(i).o(g,t),t))/n_timestep; % utility of outcomes over time steps
        end
        
    end
    
    o(:,i) = MDP(i).o(:,end); % observation
    u(:,i) = MDP(i).u(:,end); % chosen action
    
end

% Initial states and expected policies
%--------------------------------------------------------------------------
choice_prob = zeros(size(MDP(i).P,1),n_trials);
for i = 1:n_trials 
    choice_prob(:,i) = MDP(i).P;
end

t     = 1:n_trials;

% plot choices and beliefs about choices
subplot(2,1,1)

imagesc([1 - choice_prob]); colormap(cols) , hold on
chosen_action = u; plot([chosen_action],'.','MarkerSize',MarkerSize,'Color',col{1})

for i = 1:max(o(1,:))
    j = find(o(1,:) == i);
    plot(t(j),ones(1,length(t(j))),'.','MarkerSize',MarkerSize,'Color',col{i})
end


title('Action Probabilities and Chosen Actions')
xlim([-1,n_trials+1])
set(gca, 'XTick', [0:n_trials]), 
set(gca, 'YTick', [1:4]), set(gca, 'YTickLabel', {'Wins','Bandit 1','Bandit 2','Bandit 3'})
xlabel('Trial'),ylabel('Policy')

hold off

subplot(4,1,3)

col   = {[0.4660, 0.4740, 0.1880], ...  % green
         [0.4660, 0.6740, 0.1880], ...  % green
         [0.4660, 0.8740, 0.1880], ...  % green
         [0.5350, 0.1780, 0.2840], ...  % red
         [0.7350, 0.1780, 0.2840], ...  % red
         [0.9350, 0.1780, 0.2840]};     % red

a_con_1 = zeros(2,n_trials+1);
a_fig_1 = zeros(2,n_trials+1);
a_con_2 = zeros(2,n_trials+1);
a_fig_2 = zeros(2,n_trials+1);
a_con_3 = zeros(2,n_trials+1);
a_fig_3 = zeros(2,n_trials+1);

a_con_1(:,1) = [prior_a; prior_a];
a_fig_1(:,1) = [prior_a; prior_a];
a_fig_1(:,1) = a_fig_1(:,1)/sum(a_fig_1(:,1));
a_con_2(:,1) = [prior_a; prior_a];
a_fig_2(:,1) = [prior_a; prior_a];
a_fig_2(:,1) = a_fig_2(:,1)/sum(a_fig_2(:,1));
a_con_3(:,1) = [prior_a; prior_a];
a_fig_3(:,1) = [prior_a; prior_a];
a_fig_3(:,1) = a_fig_3(:,1)/sum(a_fig_3(:,1));

for i = 1:n_trials
    a_con_1(:,i+1) = MDP(i).a{1}(2:3,2);
    a_fig_1(:,i+1) = MDP(i).a{1}(2:3,2);
    a_fig_1(:,i+1) = a_fig_1(:,i+1)/sum(a_fig_1(:,i+1));
    
    a_con_2(:,i+1) = MDP(i).a{1}(2:3,3);
    a_fig_2(:,i+1) = MDP(i).a{1}(2:3,3);
    a_fig_2(:,i+1) = a_fig_2(:,i+1)/sum(a_fig_2(:,i+1));
    
    a_con_3(:,i+1) = MDP(i).a{1}(2:3,4);
    a_fig_3(:,i+1) = MDP(i).a{1}(2:3,4);
    a_fig_3(:,i+1) = a_fig_3(:,i+1)/sum(a_fig_3(:,i+1));    
end

subplot(2,1,2)

plot(a_con_1(1,:),'.','MarkerSize',12,'Color',col{1}), hold on
plot(a_con_1(1,:),':','Color',col{1})

plot(a_con_1(2,:),'.','MarkerSize',12,'Color',col{4})
plot(a_con_1(2,:),':','Color',col{4})

plot(a_con_2(1,:),'.','MarkerSize',12,'Color',col{2}), hold on
plot(a_con_2(1,:),':','Color',col{2})

plot(a_con_2(2,:),'.','MarkerSize',12,'Color',col{5})
plot(a_con_2(2,:),':','Color',col{5})

plot(a_con_3(1,:),'.','MarkerSize',12,'Color',col{3}), hold on
plot(a_con_3(1,:),':','Color',col{3})

plot(a_con_3(2,:),'.','MarkerSize',12,'Color',col{6})
plot(a_con_3(2,:),':','Color',col{6})

title('Concentration Parameters')
xlim([0,n_trials+2])
ylim([0,max(max(a_con_1))+1])
xlabel('Trial'),ylabel('Value')
set(gca, 'XTick', [1:n_trials+1])
set(gca, 'XTickLabel', [0:n_trials])

sgtitle(ggtitle) 

hold off


