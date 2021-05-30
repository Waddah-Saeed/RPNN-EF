% Ridge Polynomial Neural Network with Error Feedback
% Author: Waddah Waheeb (waddah.waheeb@gmail.com)

% For more detail about the model, please refer to the following article: 
% Ridge Polynomial Neural Network with Error Feedback for Time Series Forecasting
% Link: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0167248

% If you use it, please cite it as follows:
% @article{waheeb2016ridge,
%   title={Ridge polynomial neural network with error feedback for time series forecasting},
%   author={Waheeb, Waddah and Ghazali, Rozaida and Herawan, Tutut},
%   journal={PloS one},
%   volume={11},
%   number={12},
%   pages={e0167248},
%   year={2016},
%   publisher={Public Library of Science}
% }
% or
% Waheeb, W., Ghazali, R., & Herawan, T. (2016). Ridge polynomial neural network with error feedback for time series forecasting. PloS one, 11(12), e0167248.

%% STEP 1: Load & Prepare time series data

    % Inputs: 
    %            data: univariate time series
    %       test_size: the size of out-of-sample set
    %   lower & upper: lower and upper values to scale the time series 
    %                  using min_max method
    %            lags: number of past values to be used to forecast 
    %                  the future values
    
    % Outputs:
    %              ts: in which variables about the original and scaled time
    %                  series are stored
    %             ann: in which inputs_train, targets_train, inputs_test &
    %                  targets_test are stored   

    load('data_example.mat');
    disp('Data has been loaded.');
    
    train_end = 297; % training set size
    inputs_train = data_inputs(1:train_end,:);
    targets_train = data_targets(1:train_end,:);

    inputs_test = data_inputs(train_end+1:end,:);
    targets_test = data_targets(train_end+1:end,:);
    disp('Data has been splitted and ready for training.'); 
    
%% STEP 2: Train RPNN-EF

    rng(1,'twister');    
    ann.input_nodes = size(inputs_train,2) + 1;  % 3 lags + 1 error feedback
   
    % for better forecasting performance you need to try different values at least for these three parameters
    ann.lr=0.1;                 %	Learning rate value (try for example [0.01, 0.03, 0.1, 0.3, 1])
    ann.mom=0.9;                 %	Momentum value (try for example [0.9, 0.8, 0.7, ...])
    ann.r=0.00001;               %   Threshold to increase another pi-sigma network of increasing order (try for example [0.00001, 0.0001, 0.001, 0.01, 0.1])
    ann.dec_r=0.01;              %   Decrease factor of r (try for example [0.05, 0.1, 0.2])

    ann.dec_lr = 0.8;           %   Decrease factor of lr
    ann.max_epoch= 3000;        % 	Maximum number of epochs
    ann.max_order=5;            %   Maximum order of the network
    ann.min_err=0.00001;        %   Threshold to stop the whole training
    ann.factor_reduction=1e-8;  %   Stop training if there is no reduction in error by a factor of at least 1 - factor_reduction
    
    ann.repeats=3;             %   Number of networks to train

    disp('Start training...');
    
    [ results_train, net, ann_new ] = RPNNEF.fit( ann, inputs_train, targets_train );   % train the networks
    
    disp('Training has been completed.');
    
    % forecasts using mean and median combination
    results_train.network_outputs_combined = RPNNEF.combine_forecasts(results_train.network_outputs);
    
    % check how good is forecasting performance using training set 
    perf_training = RPNNEF.performance( results_train.network_outputs_combined, targets_train );
    
    disp('************************************');
    disp('----------Training error-RMSE----------');
    disp(['mean: ',num2str(perf_training.RMSE(1,1))]);
    disp(['median: ',num2str(perf_training.RMSE(1,2))]);
    disp('----------Training error-MAE----------');
    disp(['mean: ',num2str(perf_training.MAE(1,1))]);
    disp(['median: ',num2str(perf_training.MAE(1,2))]);
    disp('----------Training error-NMSE----------');
    disp(['mean: ',num2str(perf_training.NMSE(1,1))]);
    disp(['median: ',num2str(perf_training.NMSE(1,2))]);    
    disp('************************************');
    
    figure(1);
    xx = 1:1:length(targets_train);
    plot(xx,targets_train,'-k','LineWidth',1);
    hold on;
    plot(xx,results_train.network_outputs_combined(:,1),'-r','LineWidth',1);
    plot(xx,results_train.network_outputs_combined(:,2),'-g','LineWidth',1);
    xlabel('Time');
    ylabel('Time series values');
    title('Training forecasting');
    legend('Original','Fitting using mean combination','Fitting using median combination');
    hold off;
    
%% STEP 3: Forecast
    forecast_horizon = 10;
    results_test = RPNNEF.forecast( ann, net, inputs_test, targets_test, forecast_horizon );
    
    % check how good is forecasting performance using out-of-sample set  
    points = size(results_test.combines,1);
    
    perf_test = RPNNEF.performance( results_test.combines, targets_test(1:points,1) );
    
    disp('************************************');
    disp('----------Out-of-sample error-RMSE----------');
    disp(['mean: ',num2str(perf_test.RMSE(1,1))]);
    disp(['median: ',num2str(perf_test.RMSE(1,2))]);
    disp('----------Out-of-sample error-MAE----------');
    disp(['mean: ',num2str(perf_test.MAE(1,1))]);
    disp(['median: ',num2str(perf_test.MAE(1,2))]);
    disp('----------Out-of-sample error-NMSE----------');
    disp(['mean: ',num2str(perf_test.NMSE(1,1))]);
    disp(['median: ',num2str(perf_test.NMSE(1,2))]);    
    disp('************************************');
    
    figure(2);
    xx = 1:1:points;
    plot(xx,targets_test(1:points,1),'-k','LineWidth',1);
    hold on;
    plot(xx,results_test.combines(:,1),'-r','LineWidth',1);
    plot(xx,results_test.combines(:,2),'-g','LineWidth',1);
    xlabel('Time');
    ylabel('Time series values');
    title('Out-of-sample forecasting');
    legend('Original','Forecasts using mean combination','Forecasts using median combination');
    hold off;
