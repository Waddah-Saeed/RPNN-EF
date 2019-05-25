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

    % suppose the univariate time series is called 'data'
    load('example.mat');
    test_size = 300;    % the size of out-of-sample set
    ts = TS.prepare(data, test_size);   % prepare time series based on the given test_size
    
    lower=0.2;
    upper=0.8;
	ts_scaled = TS.scale(ts,lower,upper);   % scale time series based on the given lower and upper values
    
    lags=3;
    ts_scaled = TS.createlags(ts_scaled,lags);  % construct samples based on the given lags
    [ann] = TS.split(ts_scaled);    % split the data into training and out-of-sample sets

    
%% STEP 2: Train RPNN-EF & do 1-step forecasting

    % You can escape STEP 1 if your training and out-of-sample(i.e., test) sets are ready 
    % just pass them as follows:
        % ann.inputs_train = (your training inputs [N*lags]);
        % ann.targets_train = (your training targets [N*1]);
        % ann.inputs_test = (your out-of-sample inputs [M*lags]);
        % ann.targets_test = (your out-of-sample targets [M*1]);
        % lags = ;
    
    ann.input_nodes = lags + 1;  % 3 lags + 1 error feedback
   
    % for better forecasting performance you need to try different values at least for these three parameters
    ann.lr=0.3;                 %	Learning rate value (try for example [0.01, 0.03, 0.1, 0.3, 1])
    ann.r=0.0001;               %   Threshold to increase another pi-sigma network of increasing order (try for example [0.00001, 0.0001, 0.001, 0.01, 0.1])
    ann.dec_r=0.01;              %   Decrease factor of r (try for example [0.05, 0.1, 0.2])

    ann.dec_lr = 0.8;           %   Decrease factor of lr
    ann.max_epoch= 3000;        % 	Maximum number of epochs
    ann.max_order=5;            %   Maximum order of the network
    ann.min_err=0.00001;        %   Threshold to stop the whole training
    ann.factor_reduction=1e-8;  %   Stop training if there is no reduction in error by a factor of at least 1 - factor_reduction
    
    ann.repeats=3;             %   Number of networks to train
    
    [ results_train, net, ann_new ] = RPNNEF.training( ann );   % train the networks
    
    % forecasts using mean and median combination
    results_train.training_forecasts_scaled = RPNNEF.combine_forecasts(results_train.forecasts_train);

    % forecast 1-step ahead using out-of-sample set
    results_test = RPNNEF.forecast( ann, net );
    
%% STEP 3: De-scale
    % if you scale the data, you need to return it to its original scale,
    % otherwise, uncomment these
    %   results_train.training_forecasts = results_train.training_forecasts_scaled;
    %   results_test.forecasts = results_test;
    
    % return the forecasts to the original scale
    results_train.training_forecasts = TS.descale(results_train.training_forecasts_scaled, lower, upper, ts.minn, ts.maxx);
    
    % return the forecasts to the original scale
    results_test.forecasts = TS.descale(results_test.combines, lower, upper, ts.minn, ts.maxx);
    
%% STEP 4: Print & plot the forecasting performance

    % if you didn't use STEP 1, assign your original training and testing
    % sets to these variables
    OriginalTrainingSeries = ts.ts_train(lags+1:end,1);
    OriginalTestingSeries = ts.ts_test;
    
    % check how good is forecasting performance using training set 
    perf_training = RPNNEF.performance( results_train.training_forecasts, OriginalTrainingSeries );
 
    % check how good is forecasting performance using out-of-sample set     
    perf_test = RPNNEF.performance( results_test.forecasts, OriginalTestingSeries );

%% STEP 5: Print results

    disp('************************************');
    disp('-----------Training error----------');
    disp(['RMSE (mean): ',num2str(perf_training.RMSE(1,1))]);
    disp(['RMSE (median): ',num2str(perf_training.RMSE(1,2))]);
    disp('************************************');

    disp('************************************');
    disp('--------Out-of-sample error--------');
    disp(['RMSE (mean): ',num2str(perf_test.RMSE(1,1))]);
    disp(['RMSE (median): ',num2str(perf_test.RMSE(1,2))]);
    disp('************************************');

    figure(1);
    xx = 1:1:length(OriginalTestingSeries);
    plot(xx,ts.ts_test,'-k','LineWidth',1);
    hold on;
    plot(xx,results_test.forecasts(:,1),'-r','LineWidth',1);
    xlabel('Time');
    ylabel('Time series values');
    title('Out-of-sample forecasting');
    legend('Original','Forecasts using mean combination', 1);
    hold off;

    figure(2);
    plot(xx,ts.ts_test,'-k','LineWidth',1);
    hold on;
    plot(xx,results_test.forecasts(:,2),'-b','LineWidth',1);
    xlabel('Time');
    ylabel('Time series values');
    title('Out-of-sample forecasting');
    legend('Original','Forecasts using median combination', 1);
    hold off;