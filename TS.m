classdef TS
       
    methods(Static)
        
        function [ts] = prepare(data, test_size)
            
            ts.ts = data;   % store the original time series data
            
            ts.ts_size= length(data);   % the size of time series
            ts.ts_test_size=test_size;  % the size of time series (out-of-sample set)
            ts.ts_train_size=ts.ts_size-ts.ts_test_size;    % the size of time series (training set)
            
            ts.ts_train = ts.ts(1:ts.ts_train_size,1);  % training set
            ts.ts_test = ts.ts(ts.ts_train_size + 1:end ,1); % out-of-sample set          

            ts.minn = min(ts.ts_train); % minimum value in the training set
            ts.maxx = max(ts.ts_train); % maximum value in the training set
            
        end
           
        function [n_ts] = scale(ts,lower, upper)
            % scale time series based on min_max method
            
            n_ts.lower= lower;
            n_ts.upper= upper;
            mul = upper - lower;
            
            onevr = ones(ts.ts_train_size,1);
            n_ts.ts_train = (mul*(ts.ts_train-ts.minn*onevr)./((ts.maxx-ts.minn)*onevr)) + lower;
            
            onevr = ones(ts.ts_test_size,1);
            n_ts.ts_test = (mul*(ts.ts_test-ts.minn*onevr)./((ts.maxx-ts.minn)*onevr)) + lower;
            
            n_ts.ts = [n_ts.ts_train; n_ts.ts_test];
            n_ts.ts_size=ts.ts_size;
            n_ts.ts_test_size=ts.ts_test_size;
        end
        
        function [ts] = createlags(ts,lags)
            % construct samples based on the given lags
            samples=ts.ts_size-lags;

            ts.lags=lags;
            ts.ts_train_size = samples - ts.ts_test_size;
            
            sze=lags-1;
            ts.data_inputs=zeros(samples,lags);
            ts.data_targets=zeros(samples,1);

            for i=1:samples
                ts.data_inputs(i,:)= ts.ts(i:i+sze,1)';
                ts.data_targets(i,:)= ts.ts(i + lags,1);
            end
        end
        
        function [data] = split(ts)
            % split the data into training and out-of-sample sets
            
            data.inputs_train = ts.data_inputs(1:ts.ts_train_size,:);
            data.targets_train = ts.data_targets(1:ts.ts_train_size,:);

            data.inputs_test = ts.data_inputs(ts.ts_train_size+1:end,:);
            data.targets_test = ts.data_targets(ts.ts_train_size+1:end,:);
            
        end

        function [new_data] = descale(data, lower, upper, minn, maxx)
            % return to the original scale
            
            [R,C] = size(data);
            onevr = ones(R,1);
            mul = upper - lower;
            new_data= zeros(R,C);
            
            for i=1:C
                new_data(:,i) = (data(:,i)-lower)/mul;
                new_data(:,i) = new_data(:,i).*((maxx-minn)*onevr) + minn*onevr;
            end

        end

    end
    
end