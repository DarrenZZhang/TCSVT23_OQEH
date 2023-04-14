function [] = demo_eval(n_bits, dataset, prefix, epoch, inter)
%    n_bits = nbit;
%    dataset = dset;
    addpath('utils/');

%     n_bits  = 32;
%     dataset = 'coco';
%     prefix  = '0906';
%      iter   = epoch;

     for iter = inter: inter: epoch

        % load mat
        fprintf('Load hashcodes and labels of %s.\n', dataset);
        hashcode_path = sprintf('../hashcode/HASH_%s_%dbits_%d.mat', dataset, n_bits, iter);
        load (hashcode_path);
        fprintf('End Load hashcodes and labels of %s.\n', dataset);

        % get label
        trn_label = double(L_db);
        tst_label = double(L_te);
        cateRetriTest = sign(double(trn_label) * double(tst_label')) == 1;
   
        % get hash code
        B = compactbit(retrieval_B > 0);
        tB = compactbit(val_B > 0);
        hamm = hammingDist(tB, B)';
        [~, hammRetriTest] = sort(hamm, 1);

        % eval
        [precision, recall, map] = fast_PR_MAP(int32(cateRetriTest), int32(hammRetriTest));

        clear hamm;
        clear cateRetriTest;
    
       % for draw
        draw_pre_topk = zeros(20);
        draw_rec_topk = zeros(20);
        draw_map_topk = zeros(20);
        for i=50:50:1000
            draw_pre_topk(floor(i/50)) = precision(i);
            draw_rec_topk(floor(i/50)) = recall(i);
            draw_map_topk(floor(i/50)) = map(i);
        end

        map_mean = map(1, size(hammRetriTest, 1));
        clear hammRetriTest;

        % topk results
        topk = 1000;
        map_topk       = map(topk);
        precision_topk = precision(topk);
        recall_topk    = recall(topk);

        % the final results
        result.map_mean       = map_mean;
        result.(['map_', num2str(topk)])      = map_topk;
        result.(['precision_', num2str(topk)]) = precision_topk;
        result.(['recall_', num2str(topk)])    = recall_topk;

        fprintf('--------------------Evaluation: mAP@1000-------------------\n')
        fprintf('mAP@1000 = %04f\n', result.map_1000);

        % save
        result_name = ['../results/' prefix '_' dataset '_' num2str(n_bits) 'bits_' num2str(iter) '_' datestr(now,30) '.mat'];
        save(result_name, 'precision', 'recall', 'map', 'result', ...
            'draw_pre_topk', 'draw_rec_topk', 'draw_map_topk');
end






