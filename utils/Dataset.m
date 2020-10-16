classdef Dataset
    
    properties
        name;
        task;

        n_brunch = 0
        n_agents = 0

        % for normal centralized data
        fea = []
        gnd = []

        n_fea = 0
        n_smpl = 0
        
        % for centralized labeled data
        fea_l = []
        gnd_l = []
        n_smpl_l = 0
        
        % for centralized unlabeled data
        fea_u = []
        gnd_u = []
        n_smpl_u = 0
        
        % for normal distributed data
        fea_d = []
        gnd_d = []

        n_fea_d = 0
        n_smpl_d = 0
        
        % for distributed labeled data
        fea_d_l = []
        gnd_d_l = []
        n_smpl_d_l = 0
        
        % for distributed unlabeled data
        fea_d_u = []
        gnd_d_u = []
        n_smpl_d_u = 0

        % for centralized data on hierarchical structure
        fea_c_h = [] % (n_brunch, n_smpl_c_h, n_fea_c_h)
        gnd_c_h = []

        n_fea_c_h = 0
        n_smpl_c_h = 0

        % for distributed data on distributed hierarchical structure
        fea_d_h = [] %(n_agents, n_brunch, n_smpl_d_h, n_fea_d_h)
        gnd_d_h = [] %(n_agents, n_smpl_d_h)

        n_fea_d_h = 0
        n_smpl_d_h = 0

        partition;
    end
    
    methods
        function obj = Dataset(fea, gnd, task, name)
            obj.name = name;
            obj.fea = fea;
            obj.task = task;
            obj.gnd = gnd;
            obj.n_fea = size(fea, 2);
            obj.n_smpl = size(fea, 1);
        end

        function obj = set_partition(obj, partition)
            obj.partition = partition;
        end

        function obj = get_data_c_h(obj, n_brunch)
            %
                %todo:split dataset into n_brunchs based on features
                %:param n_brunch: the number of hierarchy brunches
                %:return: 1st fold datasets for run by default or specified n fold runabel datasets
            %
            obj.n_fea_c_h = floor(obj.n_fea / n_brunch);
            obj.n_smpl_c_h = obj.n_smpl;
            obj.n_brunch = n_brunch;

            h_fea = zeros(n_brunch, obj.n_smpl_c_h, obj.n_fea_c_h);
            for i = 1:n_brunch
                h_fea(i, :, :) = obj.fea(:, obj.n_fea_c_h*(i-1) +1:obj.n_fea_c_h * i);
            end

            obj.fea_c_h = h_fea;
            obj.gnd_c_h = obj.gnd;
        end

        function obj = get_data_d_h(obj, n_brunch, n_agents)
            %
                %todo:distribute dataset into n_agents based on sample
                %level
                %:param n_agents: the number of distributed agents
                %:param n_brunch: the number of hierarchy brunches
                %:return: 1st fold datasets for run by default or specified n fold runabel datasets
            %"""
            obj = obj.get_data_c_h(n_brunch);
            obj.n_smpl_d_h = floor(obj.n_smpl / n_agents);
            obj.n_fea_d_h = obj.n_fea_c_h;

            obj.n_agents = n_agents;

            hd_fea = zeros(n_agents, n_brunch, obj.n_smpl_d_h, obj.n_fea_d_h);
            hd_gnd = zeros(n_agents, obj.n_smpl_d_h);

            for i = 1:n_agents
                hd_fea(i, :, :, :) = obj.fea_c_h(:, obj.n_smpl_d_h * (i-1) +1:obj.n_smpl_d_h * i, :);
                gnd_c_h_tmp = obj.gnd_c_h(:, 1);
                hd_gnd(i, :) = gnd_c_h_tmp(obj.n_smpl_d_h * (i-1) +1:obj.n_smpl_d_h * i);
            end
            obj.fea_d_h = hd_fea;
            obj.gnd_d_h = hd_gnd;
        end
        
        function obj = get_data_d_c(obj, n_agents)
            %
                %todo:distribute centralized dataset into n_agents based on sample
                %level
                %:param n_agents: the number of distributed agents
                %:param n_brunch: the number of hierarchy brunches
                %:return: 1st fold datasets for run by default or specified n fold runabel datasets
            %"""
            
            obj.n_smpl_d = floor(obj.n_smpl / n_agents);
            obj.n_fea_d = obj.n_fea;

            obj.n_agents = n_agents;

            hd_fea = zeros(n_agents, obj.n_smpl_d, obj.n_fea_d);
            hd_gnd = zeros(n_agents, obj.n_smpl_d);

            for i = 1:n_agents
                hd_fea(i, :, :) = obj.fea( obj.n_smpl_d * (i-1) +1:obj.n_smpl_d * i, :);
%                 gnd_c_h_tmp = obj.gnd(:, 1);
                hd_gnd(i, :) = obj.gnd(obj.n_smpl_d * (i-1) +1:obj.n_smpl_d * i);
            end
            obj.fea_d = hd_fea;
            obj.gnd_d = hd_gnd;
        end

        function [train_data, test_data] = get_kfold_data_dh(obj, n_brunch, n_agents, fold_idx)
        %
            %todo:generate hierarchical distributed training dataset and test dataset by k-folds
            %:param n_agents: the number of distributed agents
            %:param n_brunch: the number of hierarchy brunches
            %:param fold_idx:
            %:return: 1st fold datasets for run by default or specified n fold runabel datasets
        %
            [train_data, test_data] = get_kfold_data(obj, fold_idx);

            % get all useful data type
            train_data = train_data.get_data_d_h(n_brunch, n_agents);
            test_data = test_data.get_data_d_h(n_brunch, n_agents);
        end
        
        function [train_data, test_data] = get_kfold_data_d(obj, n_agents, fold_idx)
        %
            %todo:generate distributed training dataset and test dataset by k-folds
            %:param n_brunch: the number of hierarchy brunches
            %:param fold_idx:
            %:return: 1st fold datasets for run by default or specified n fold runabel datasets
        %
            [train_data, test_data] = get_kfold_data(obj, fold_idx);

            % get all useful data type
            train_data = train_data.get_data_d_c(n_agents);
            test_data = test_data.get_data_d_c(n_agents);
        end
        
        function [train_data, test_data] = get_kfold_data(obj, fold_idx)
        %
            %todo:generate training dataset and test dataset by k-folds
            %:param n_brunch: the number of hierarchy brunches
            %:param fold_idx:
            %:return: 1st fold datasets for run by default or specified n fold runabel datasets
        %
            if nargin == 2
                obj.partition.current_fold = fold_idx;
            end
            [train_idx, test_idx] = obj.partition.get_data_idx();

            train_name = sprintf('%s_train', obj.name);
            test_name = sprintf("%s_test", obj.name);

            % if the dataset is like a eeg data, which has trails hold sample blocks
            if size(size(obj.fea), 2) == 3
                % reform training dataset
                train_data = Dataset(obj.fea(train_idx, :, :), obj.gnd(train_idx, :), obj.task, train_name);
                gnd_train = [];
                fea_train = [];
                for ii = 1:size(train_data.gnd, 1)
                    fea_train = [fea_train; train_data.fea(ii)];
                    size_smpl_ii = size(train_data.fea(ii),1);
                    gnd_train_tmp = repeat(train_data.gnd(ii), size_smpl_ii, 1);
                    gnd_train = [gnd_train; gnd_train_tmp];
                end
                train_data = Dataset(fea_train, gnd_train, train_data.task, train_data.name);

                % reform test dataset
                test_data = Dataset(obj.fea(test_idx, :), obj.gnd(test_idx, :), obj.task, test_name);
                gnd_test = [];
                fea_test = [];
                for ii = 1:size(test_data.gnd, 1)
                    fea_test = [fea_test; test_data.fea(ii)];
                    size_smpl_ii = size(test_data.fea(ii), 1);
                    gnd_test_tmp = repeat(test_data.gnd(ii), size_smpl_ii, 1);
                    gnd_test = [gnd_test; gnd_test_tmp];
                end
                test_data = Dataset(fea_test, gnd_test, test_data.task, test_data.name);
            else
                train_data = Dataset(obj.fea(train_idx, :), obj.gnd(train_idx, :), obj.task, train_name);
                test_data = Dataset(obj.fea(test_idx, :), obj.gnd(test_idx, :), obj.task, test_name);
            end

%             % normalize data
%             fea_all = [train_data.fea; test_data.fea];
%             fea_normalize = mapminmax(fea_all', -1, 1)';
%             train_data.fea = fea_normalize(1:train_data.n_smpl, :);
%             test_data.fea = fea_normalize(train_data.n_smpl+1:end, :);

        end

        function obj = set_shuffle(obj, shuffle)
            obj.shuffle = shuffle;
        end
        
	end 
        
        
    
end

