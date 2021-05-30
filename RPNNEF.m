classdef RPNNEF
    % Ridge Polynomial Neural Network with Error Feedback
    % Author: Waddah Waheeb & Rozaida Ghazali

    % For more detail about the model, please refer to the following article: 
    % Ridge Polynomial Neural Network with Error Feedback for Time Series Forecasting
    % Link: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0167248
    properties
    end
    
    methods(Static)
        
        function [ results, net, ann ] = fit( ann, inputs_train, targets_train )
            % train networks

            results.time = -1*ones(ann.repeats,ann.max_order);                % time
            results.mse_train = 100*ones(ann.repeats,ann.max_order);          % MSE on training data
            results.epoch= -1*ones(ann.repeats,ann.max_order);                % epochs
            results.err_train=zeros(ann.repeats,ann.max_epoch);               % training error
            results.forecasts_train=[];                                       %forecasts (training)

            net.weights_initial=cell(ann.repeats,ann.max_order);              % initial weights
            net.biases_initial=cell(ann.repeats,ann.max_order);               % initial biases 
            net.weights_final=cell(ann.repeats,ann.max_order);                % final weights
            net.biases_final=cell(ann.repeats,ann.max_order);                 % final biases

            ann.factor=1/(1-ann.factor_reduction);      % calculate the value to stop training if there is no error reduction
            ann.output_node = 1;      % 1 output node
            ann.initFeedback=0.5;     % initial error feedback value
            
            for n=1:ann.repeats

                order =1;
                prev_err = 0.5;     % initial previous error
                all_tr_error = [];  % training errors
                curr_err = 10;      % just initialize with big value (for training)
                epoch = 1;
                SH = 1;
                factor_flag = 0;    % to control error reduction stop criterion
                
                % create the ridge and initialize its weights and biases
                NN1 = RPNNEF.create_ridge(ann.input_nodes,ann.initFeedback);

                while ((curr_err > ann.min_err) && (order <=ann.max_order) && (epoch < ann.max_epoch) && (factor_flag==0))

                    start_time=tic;

                    % create the pi-sigma and initialize its weights and biases
                    NN2 = RPNNEF.create_psnn(ann.input_nodes,order,ann.lr,SH,ann.mom);

                    % store initial weights & biases
                    net.weights_initial{n,order}=NN2.W1;
                    net.biases_initial{n,order}=NN2.B1;

                    % combine weights & biases
                    if order==1
                        NN1.W1 = NN2.W1;
                        NN1.B1 = NN2.B1;
                    else
                        NN1.W1 = [NN1.W1,NN2.W1];
                        NN1.B1 = [NN1.B1,NN2.B1];
                    end

                    disp(' ');
                    disp(['Run : ',num2str(n), ', RPNN-EF of order : ',num2str(order)]);
                    
                    % start training
                    [NN2,NN1, forecast_TR, curr_err, epoch, prev_err,all_tr_error,factor_flag] = RPNNEF.train(NN2,NN1,ann, inputs_train, targets_train, order, prev_err, epoch ,all_tr_error); 

                    % store results after training for the given network order
                    results.time(n,order) = toc(start_time);
                    results.epoch(n,order) = epoch; 
                    results.mse_train(n,order) = curr_err;
                    %results.forecasts_train=[results.forecasts_train,forecast_TR];

                    results.network_outputs(:,n) = forecast_TR;
                    
                    net.weights_final{n,order}=NN1.W1;
                    net.biases_final{n,order}=NN1.B1;

                    % update training parameters
                    ann.lr = ann.lr * ann.dec_lr;
                    ann.r = ann.r * ann.dec_r;
                    SH = order + NN2.store_hidden;
                    order = order + 1;
                    
                end
                
                net.order(n)= order-1;
                results.err_train(n,1:length(all_tr_error))=all_tr_error;
                net.NN{n}=NN1;
                
            end
            
        end
        function [ results ] = combine_forecasts( outputs )

            % mean and median combination
            sze=size(outputs,2);
            if(mod(sze,2)~=1) 
                sze=sze-1;
                disp('The last simulation results will be ignored when calculating the median.');
            end
            results(:,1)=mean(outputs,2);
            results(:,2)=median(outputs(:,1:sze),2);
            
        end        
        function [ results ] = performance( forecasts, targets )
            % forecasting performance
            
            results.Error = targets - forecasts;
            SE = power(results.Error,2);
            results.MSE= mean(SE);
            results.RMSE= sqrt(results.MSE);
            results.MAE= mean(abs(results.Error));

            mean_targets = mean(targets);
            summ = sum(power(targets-mean_targets,2));
            results.NMSE = sum(SE) / summ;
            
        end       
        function [ results ] = forecast( ann, net, inputs_test, targets_test, forecast_horizon  )
            
            % 1-step forecasts based on mean and median combination 
            results.fcasts=[];
            results.err_input_test_row=[];
            
            for n=1:ann.repeats

                if (forecast_horizon == 1)
                    [forecast_TEST, err_input_test_row] = RPNNEF.one_step(net.NN{1,n}, inputs_test, targets_test, net.order(1,n));
                else
                    [forecast_TEST, err_input_test_row] = RPNNEF.multi_step(net.NN{1,n}, net.order(1,n), forecast_horizon, inputs_test(1,:));
                end
                
                results.fcasts=[results.fcasts,forecast_TEST];
                results.err_input_test_row=[results.err_input_test_row,err_input_test_row];        

            end
            
            results.combines= RPNNEF.combine_forecasts( results.fcasts );
        end
        
    end
    
    methods(Access = private, Static)
        
        function [NN1] = create_ridge(numIn, initFeedback)
            % initialization
            
            numHidden=1;

            r=power((6/(numIn+1)),1/4);
            NN1.W1 = (-r) + (r+r).*rand(numIn,numHidden);
            NN1.B1 = zeros(1,numHidden);
            
            NN1.W1_update = zeros(numIn,numHidden);
            NN1.B1_update = zeros(1,numHidden);

            NN1.Z = initFeedback;
        end
        function [NN2] = create_psnn(numIn,numHidden,LR,SH,momentum)
            % initialization
            
            r=power((6/(numIn+1)),1/4);
            NN2.W1 = (-r) + (r+r).*rand(numIn,numHidden);            
            NN2.B1 =zeros(1,numHidden);
            
            NN2.W1_update = zeros(numIn,numHidden);
            NN2.B1_update = zeros(1,numHidden);

            NN2.W1_delta = zeros(numIn,numHidden);
            NN2.B1_delta = zeros(1,numHidden);

            NN2.LR = LR;
            NN2.MOM = momentum;
            
            NN2.P_previous = zeros(numIn,SH);  %Previous state (the dynamic system variable)
            NN2.PBIAS_previous = zeros(1,SH);

            NN2.store_hidden = SH;

        end
        function [NN2,NN1, network_outputs,training_error, epoch, prev_err, all_tr_error, factor_flag] = train(NN2,NN1,ann, inputs_train, targets_train, order, prev_err, epoch , all_tr_error)
            % train a network model
            
            NN2.epoch  = epoch;
            factor_flag = 0;
            len = length(inputs_train);
            
            % because adding a new higher order leads usually in increasing error 
            % therefore when this variable equals 1 means no need to check
            % error reduction
            new_order=1;    

            while (epoch < ann.max_epoch)
                
                tr_error = 0;
                network_outputs = zeros(len,1); %PRE-ALLOCATION
                
                for i = 1 : len

                    X = inputs_train(i,:);
                    Y = targets_train(i,:);

                    [NN1,NN2] = RPNNEF.feedforward(X,NN1,NN2,order,Y);
                    error_1 = (Y - NN1.output_signal);

                    [NN1, NN2] = RPNNEF.backpropagate(error_1,NN1,NN2, epoch,i);

                    network_outputs(i)=NN1.output_signal;
                    tr_error = tr_error + power(error_1,2.0) ; %SSE

                    [NN1, NN2] = RPNNEF.update(NN1,NN2,order);  

                end

                training_error = tr_error / len; % MSE
                all_tr_error = [all_tr_error; training_error];  % all training error
                
                if(training_error <= ann.min_err)
                    break;  % exit from while loop
                end 
                
                if(epoch > 2)
                    val = all_tr_error(epoch,1)/all_tr_error(epoch-1,1);
                    if(val>ann.factor && new_order~=1)
                        disp('------------------------------------------------------------------------');
                        disp('unable to reduce the error by a factor of at least 1 - factor_reduction.');
                        disp('------------------------------------------------------------------------');                        
                        factor_flag = 1;
                        break;
                    end
                end
                
                % calculate whether to increase the order of pi-sigma or not
                check_add_order = abs(training_error - prev_err)/ prev_err;
                prev_err = training_error;
                epoch = epoch + 1; 
                    
                if (check_add_order < ann.r)
                    % exit to add a higher order
                    break;
                end
                new_order=0;
            end

           disp(['Epoch = ', num2str(epoch), ', Training Error = ', num2str(training_error), ', Target Error = ', num2str(ann.min_err)]);

        end
        function [NN1,NN2] = feedforward(X,NN1,NN2,order,Y)
            % feedforward the inputs and calculate error feedback
            
            NN1.X = [X,NN1.Z];   %concatenate external inputs and context node

            NN1.hidden = (NN1.X*NN1.W1) + NN1.B1;

            SH = 0;

            for j= 1:order
                SH =  SH + 1;
                NN1.product(j)= prod(NN1.hidden(:,SH:j - 1 + SH));
                SH = (j - 1) + SH;
            end

            %OUTPUT SIGNAL FOR OUTPUT LAYER
            NN1.output = sum(NN1.product);
            NN1.output_signal = 1.0 ./ (1.0 + exp(-1*NN1.output ));
            NN1.Z = Y-NN1.output_signal;
            
            %CALCULATE OUTPUT SIGNAL DERIVATIVE
            NN1.out_derivative = (1 - NN1.output_signal).* NN1.output_signal ;   %(1 - output)*output
        end
        function [NN1, NN2] = backpropagate(error_1,NN1,NN2, epoch, i)

            % Calculate output error
            NN1.E = error_1;

            % CALCULATE OUTPUT ERROR TERM
            NN1.W1_BOY = NN1.E*NN1.out_derivative; %(1X1)   

            % CALCULATE GRADIENT FOR W1,B1
            if ((i ~= 1) || (epoch ~=  NN2.epoch))
                weights = NN2.W1_update;
            else
                weights = NN2.W1;
            end

            [row,column] = size(weights);
            SH = NN2.store_hidden;
            NN2.hidden = NN1.hidden(:,SH:end);
            prod_hidden= prod(NN2.hidden);
            
            % %involve in the latest added PSN only
            NN2.W1_delta = zeros(row,column);
            P=zeros(row,column);
            for n = 1: row
                for h = 1 :column
                    W = prod_hidden ./ NN2.hidden(h);   
                    P(n,h) = -1*NN1.out_derivative .* W .* (NN1.X(n) + (weights(end,h) .* NN2.P_previous(n,h)));% Pij = (1 - output)*output  * W * [input node   +   (wts from context to that particular summing * Pij(previous))]
                    NN2.W1_delta(n,h) = -1*NN2.LR * NN1.E * P(n,h);
                end
            end

            NN2.B1_delta=zeros(1,column);
            PB=zeros(1,column); 
            for b=1:column
                 bias = prod_hidden ./NN2.hidden(b);
                 PB(1,b) = -1*NN1.out_derivative .* bias .* (1 + (weights(end,b) .* NN2.PBIAS_previous(1,b)));
                 NN2.B1_delta(b) = -1*NN2.LR * NN1.E * PB(1,b);
            end

            NN2.P_previous = P; % store current state as a previous state (to be used in the next time step/next pattern data)
            NN2.PBIAS_previous=PB;
        end
        function [NN1, NN2] = update(NN1,NN2, order)
            % update training parameters
            
            NN2.W1_update = NN2.W1_delta + (NN2.MOM * NN2.W1_update);
            NN2.B1_update = NN2.B1_delta + (NN2.MOM * NN2.B1_update);  

            add_col = 0;
            for i = 1: order
                add_col = (i - 1) + add_col;
            end
            FENING_W = zeros(length(NN1.X) , add_col);
            FENING_B = zeros(1 , add_col);
            NN1.W1_update = [FENING_W , NN2.W1_update ];  % inserting a zero matrix of size the previous frozen PSN
            NN1.B1_update = [FENING_B, NN2.B1_update];  

            NN1.W1 = NN1.W1 + NN1.W1_update;  % then after that update the RIDGE wts
            NN1.B1 = NN1.B1 + NN1.B1_update;
             
        end
        function [forecast_test, err_feedback_values] = one_step(net, inputs_test,targets_test,order)
            % forecast the next point based on the given inputs
            len=length(inputs_test);

            forecast_test = zeros(len,1);
            err_feedback_values = zeros(len,1);
            
            for i = 1 : len

                X = inputs_test(i,:);
                Y = targets_test(i,:);
                err_feedback_values(i,:)= net.Z;
                
                net.X = [X,net.Z];
                net.hidden = (net.X*net.W1) + net.B1;
                SH = 0;

                for j= 1:order
                    SH =  SH + 1;  
                    summing = net.hidden(:,SH:j - 1 + SH);  
                    net.product(j)= prod(summing);
                    SH = (j - 1) + SH;
                end

                %OUTPUT SIGNAL FOR OUTPUT LAYER
                net.output = sum(net.product);
                net.output_signal = 1.0 ./ (1.0 + exp(-1*net.output ));

                net.Z = Y-net.output_signal;            
            
                forecast_test(i)=net.output_signal;
            end

        end
        
        function [forecast_test, err_feedback_values] = multi_step(net, order, forecast_horizon, input_test)
            % multi-step forecasts

            forecast_test = zeros(forecast_horizon,1);
            err_feedback_values = zeros(forecast_horizon,1);
            X = input_test;
            
            for i = 1 : forecast_horizon
    
                err_feedback_values(i,:)= 0;
                
                net.X = [X,net.Z];
                net.hidden = (net.X*net.W1) + net.B1;
                SH = 0;

                for j= 1:order
                    SH =  SH + 1;  
                    summing = net.hidden(:,SH:j - 1 + SH);  
                    net.product(j)= prod(summing);
                    SH = (j - 1) + SH;
                end

                %OUTPUT SIGNAL FOR OUTPUT LAYER
                net.output = sum(net.product);
                net.output_signal = 1.0 ./ (1.0 + exp(-1*net.output ));

                net.Z = 0;
                X =  [X(1,2:end) net.output_signal];
                
                forecast_test(i)=net.output_signal;
            end

        end        
    end
end
