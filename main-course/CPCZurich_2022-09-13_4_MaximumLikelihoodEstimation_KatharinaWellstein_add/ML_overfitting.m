%% Computational Psychiatry Course (CPC) 2022
%
% Talk: Maximum Likelihood
% 
% This script illustrates the overfitting problem occuring when using pure
% Maximum Likelihood Estimation (MLE). We here use the example of synthetic
% data that is generated from a low-order polynomial and fitted with
% polynomials of different order.
% 

% ----------------------------------------------------------------------
% 
% stefanf@biomed.ee.ethz.ch
%
% Author: Stefan Fraessle, TNU, UZH & ETHZ - 2021
%        Copyright 2021 by Stefan Fraessle <stefanf@biomed.ee.ethz.ch>
%
%        Amended by Jakob Heinzle & Katharina V. Wellstein - 2022
%        (wellstein@biomed.ee.ethz.ch)
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
% 
% ----------------------------------------------------------------------

% fix RNG seed
rng(1);

% clear all variables
clear all 

% close all figures
close all


% log-likelihood function for a polynomial model with Gaussian likelihood
llh = @(y,X,sigma,theta) -sum((y-X*theta).^2)/2/sigma^2-numel(y)*log(2*pi*sigma^2)/2;

% ML estimator for theta for a polynomial model with Gaussian likelihood
ml = @(y,X,sigma) (X'*X)\(X'*y);

% set parameters for synthetic data
sigma = sqrt(.001);
theta = [.3; -.1; .5];

% generate noise-free data
x = -.5:.01:.5; x = x(:);
idxTrain = 1:10:80;
    idxTrain = [11 34 37 51 55 66 80 95] ;
xTrain = x(idxTrain);
y0 = bsxfun(@power,x,0:2)*theta;

% settings
N   = 100;
Ps  = [1 2 length(idxTrain)-1];
noPs = numel(Ps);

% array for the log likelihood estimate
llhTrain = zeros(N,noPs);

% cell array for ML estimates
thetaMls = cell(noPs,N);


% 100 iterations with random noise instantiations
for n = 1:N

    % generate noisy observation with respective variance
    e = sigma*randn(size(x));
    y = y0 + e;
    yTrain = y(idxTrain);
    
    % plot the noise-free and noisy data
    if ( n == 1 )
        figure('units','normalized','outerposition',[0 0 1 1])
        clf;
        ax1Ml = subplot(2,1,1);
        plot(xTrain,yTrain,'.','Color',[0.5 0.5 0.5],'MarkerSize',50);
        hold on
        ha(1) = plot(x,y0,'k','LineWidth',2);
        ylim([0.2 0.5])
        title('data and predictions (ML)','FontSize',18);
        ylabel('$y$','Interpreter','latex','FontSize',16)
        xlabel('$x$','Interpreter','latex','FontSize',16)
        llabelsMl = {'true model'};
        box off
        subplot(2,4,5);
        bar(1:3,theta,'FaceColor',[0.6 0.6 0.6])
        title('$\theta$ (GT)','Interpreter','latex','FontSize',18)
        xlabel('model order','FontSize',16)
        ylabel('parameter value','FontSize',16)
        ylim([-0.3 0.6])
    end
    
    % iterate over polynomial orders
    for ip = 1:noPs
        
        % design matrix
        P = Ps(ip);
        X = bsxfun(@power,xTrain,0:P);
        
        % Maximum Likelihood estimates
        thetaMls{ip,n} = ml(yTrain,X,sigma);
        
        % all data points (for plotting)
        X_all = bsxfun(@power,x,0:P);
        
        % plot the predictions and estimates of the ML approach
        if ( n == 1 && ip+5 <= 8 )
            figure(1);
            subplot(2,4,ip+5);
            bar(1:P+1,thetaMls{ip,n},'FaceColor',[0.6 0.6 0.6])
            if ( ip+5 < 8 ), ylim([-0.3 0.6]), end
            title(['$\hat{\theta}$ (ML P = ' num2str(P) ')'],'Interpreter','latex','FontSize',18)
            subplot(2,1,1);
            ha(end+1) = plot(ax1Ml,x,X_all*thetaMls{ip,n},'LineWidth',2);
            ylim([0.2 0.5])
            llabelsMl{end+1} = ['ML P = ' num2str(P)]; %#ok<SAGROW>
        end
        
        % evaluate log likelihood at the ML estimates
        llhTrain(n,ip) = llh(yTrain,X,sigma,thetaMls{ip,n});
        
        % wait for button press to continue
        if ( n == 1 )
            w = waitforbuttonpress;
        end
        
    end
    
    % plot legend
    if ( n == 1 )
        figure(1)
        legend(ha,llabelsMl,'NE')
    end
    
end

 
% plot log likelihood
figure('units','normalized','outerposition',[0 0 1 1])
for ip = 1:size(llhTrain,1)
    plot(1:size(llhTrain,2),llhTrain(ip,:),'x-','Color',[0 0 1 0.1])
    hold on
end
plot(1:size(llhTrain,2),mean(llhTrain,1),'r.-','MarkerSize',12)
xlim([0 noPs+1])
title('Log-likelihood at MLE','FontSize',18);
xlabel('model order','FontSize',16)
ylabel('llh','FontSize',16)
set(gca,'xtick',1:noPs)
set(gca,'xticklabel',Ps)
axis square
box off