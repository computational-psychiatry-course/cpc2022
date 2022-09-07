% Samuel Taylor and Ryan Smith, 2021

% Simulate AAC for the provided parameter values
function [MDPs, gen_data] = TAB_sim(alpha, cr, eta_win, eta_loss, prior_a)   

    % Hard-coded trial structure for simulations (note this should be 
    % identical to the matrix in `TAB_sim.m`).
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
    
    % Number of blocks and number of trils per block
    %NB  = 20;
    NB  = 2;
    TpB = 16;
    
    MDPs = [];
    
    o_all = [];
    u_all = [];
    
    % Simulate each block separately
    for block=1:NB
        mdp = TAB_gen_model(BlockProbs(block, :), 1, alpha, cr, eta_win, eta_loss, prior_a);
        
        % Deal for all NB*TpB trials
        MDP(1:TpB) = deal(mdp);

        % Run simulation routine
        MDP  = spm_MDP_VB_X_eta2(MDP);

        % Save to file './sim.csv'
        MDPs = [MDPs; MDP'];
        
        obs = [];
        
        for o=1:TpB
            obs = [obs MDP(o).o(1,2)];
        end
        
        o_all = [o_all; obs];
        u_all = [u_all; MDP.u];
        
        clear MDP
    end
    
    % Return just the observations and actions, for fitting simulated data
    % to. This is primarily for testing recoverability: simulate with a
    % given parameter set, and run TAB_sim_fit on this data strcuture, and
    % observe the similarity of the newly fitted values to the values used
    % to create this structure in the first place.
    gen_data = struct(                ...
        'observations', o_all',       ...
        'responses', u_all'           ...
    );
end