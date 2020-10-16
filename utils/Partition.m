classdef Partition
    
    properties
        n_fold;
        train_idx_list = [];
        test_idx_list = [];
        current_fold;
    end
    
     
    methods
        function obj = Partition(n_fold)
            obj.n_fold = n_fold;
        end
                    
        function obj = partition(obj, n_smpl, is_shuffle, seed)
            if nargin<2
                is_shuffle = true;
                seed = 0;
            end
            if nargin<3
                % fix random seed
                rng(seed);
            end
            
            total_index = 1:n_smpl;
            if is_shuffle
                total_index = randperm(n_smpl);
            end
            
            n_test_smpl = floor(n_smpl / obj.n_fold);
            
            for i = 1:obj.n_fold
                test_start = n_test_smpl * (i-1) + 1;
                test_end = n_test_smpl * i;

                test_index = total_index(test_start:test_end);

                if test_start ~= 0
                    train_index_part1 = total_index(1:test_start-1);
                    train_index_part2 = total_index(test_end+1:end);
                    train_index = [train_index_part1, train_index_part2];
                else
                    train_index = total_index(test_end+1:end);
                end

                obj.train_idx_list = [obj.train_idx_list; train_index];
                obj.test_idx_list = [obj.test_idx_list; test_index];
            end
            obj.current_fold = 1;
        end

        function [train_idx, test_idx] = get_data_idx(obj)
            train_idx = obj.train_idx_list(obj.current_fold,:);
            test_idx = obj.test_idx_list(obj.current_fold,:);
        end

    end
    
end

