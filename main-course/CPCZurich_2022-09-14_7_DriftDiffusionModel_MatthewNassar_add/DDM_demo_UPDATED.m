%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                Part 4: making evidence continuous 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% now -- lets imagine we get a CONTINUOUS readout of evidence over some
% small time period (dt). That evidence has a mean value that either favors


% To discuss: 

% 0) What are parameters? How will they affect model?
% 1) Look at while loop -- How many times will it iterate?
% 2) Where is momentary evidence? 
% 3) How is it accumulated?
% 4) When will it stop accumulating?
% 5) Run model, look at output. 
% 6) change drift rate/starting point and see how it affects RT/accuracy. 


% Maybe discuss:

% 7) RTs for error versus correct trials?
% 8) what happens when we add variability to start point? 
% 9) what happens when we add variability to drift rate? 



% Set basic parameters of drift diffusion model:
A     = .005 % Drift rate
y0    =  0   % Starting point
c     = .2   % Noise (standard deviation of evidence)
z     =  10;  % Decision threshold
ndt   = 200;  % Non decision time (in ms)


% Set trial-to-trial variability parameters:
SP_noise=0; % 2
A_var   =0; % 0.01

% Set parameters of simulation 
dt    = 0.02; % how large steps should we simulate in time? 

nSims=1000
doPlots=false;


clear rt
for j =1:nSims
    
    % Initialize things that change within trial (VARIABLES):
    y = y0+normrnd(0,SP_noise); % initialize evidence:
    t=1;   % start on first timestep.

    A_trial=A+normrnd(0, A_var);
    while abs(y(t)) < z    % Loop until accumulated evidence exceeds threshould (z)
        r=randn(1); % sample a random normal variable for the random walk...
        dW=sqrt(dt).*r; % scale the variance of the variable by the size of the timestep
        dy= A_trial.*dt +c.*dW; % compute change in accumulated evidence (accumulated signal = A.*dt; accumulated noise = c.*dW)
        y(t+1)=y(t)+dy;
        t=t+1;
    end
    

% show diffusion
plotLength=800;

if doPlots==true
true_t=(1:t).*dt;
hold on
plot([0, plotLength], [0, 0], '-k')
plot(true_t, y, 'b')
ylim([-z-1, z+1])

ylabel('Evidence for correct option')
xlabel('Time')
xlim([0, plotLength])

plot([0, plotLength], [z, z], '--k')
plot([0, plotLength], [-z, -z], '--k')
end

rt(j)=t.*dt+ndt;
isAccurate(j)=y(end)>0;

end


% saveas(gcf, 'exampleDiffusion.eps', 'epsc2')
% 


figure(1)
subplot(2, 1, 1)

hist(rt(isAccurate==1), 50)
xlim([0, 3000])
ylabel('Correct trials')
set(gca, 'fontSize', 24)

subplot(2, 1, 2)

hist(rt(isAccurate==0), 50)
xlim([0, 3000])
ylabel('Error trials')

set(gca, 'fontSize', 24)
xlabel('Reaction time')






figure(2)
hold on
plot([0, 2], [.5, .5], '--k')
bar(mean(isAccurate), 'b')
ylim([0,1])
ylabel('Accuracy')
set(gca, 'fontSize', 24)





