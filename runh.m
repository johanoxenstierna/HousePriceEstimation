
addpath(genpath('HVfunctions')); 

clear; 

rng shuffle 

% clearvars -except sample inputs targets sheet_nr sheet_index name_output_file; 

% DO ANALYSIS ON SAVED DATA

rL = {}; 
rL{1, 1} = 'MAPE'; 
rL{1, 2} = 'MDAPE'; 
rL{1, 3} = 'neurons'; 
rL{1, 4} = '--'; 
rL{1, 5} = 'epochs used'; 
rL{1, 6} = 'bestEpoch'; 
rL{1, 7} = 'trainFCN';
rL{1, 8} = 'reasonForStop';
rL{1, 9} = 'cpuTime';
rL{1, 10} = 'ticToc'; 
rL{1, 11} = 'normYN'; 
rL{1, 12} = 'numberOfParams'; 
rL{1, 13} = 'trainBorderData'; 
startingIndexAfter = length(rL) + 1; 

% OPEN OUTPUT FILE
prompt = 'What should be the name of the output file?  '; 
name_output_file = input(prompt); 
% name_output_file = 'tempOutput'; 


% if exist('name_output_file','var') == 0
%     name_output_file = 'output1'; 
%     
% else
%     fid = fopen(name_output_file,'a+');
% end


% LOAD SAMPLE if it doesnt already exist in workspace
% nParameters = 0; 
% if exist('sample','var') == 0
% [numX, titlesX, rawX] = xlsread('processed25kmF2.xlsx'); 
[numX, titlesX, rawX] = xlsread('processed25kmF5.xlsx'); 


titlesX = titlesX(1,:); % otherwise it bugs 
num = numX'; titles = titlesX'; raw = rawX'; 

% check that raw is sorted properly and convert to datenum (MATLAB 2016a
% has no support for dates)

indContractdate = find(cellfun('length',regexp(titles,'t_contractdate')) == 1);

theDates = cell2mat(raw(indContractdate, 2:end)); 

if theDates(1) > theDates(2) || ...
    theDates(5) > theDates(200) || ...
    theDates(500) > theDates(501) || ...
    theDates(501) > theDates(1001) || ...
    theDates(1298) > theDates(end) 
    throw(MException('Data is not sorted correctly')); 
end



% find indxes
indTargetsRecounted = find(cellfun('length',regexp(titles,'recounted_price')) == 1);
indTargetsNotRecounted = find(cellfun('length',regexp(titles,'t_price')) == 1);
targetsRecounted = num(indTargetsRecounted,:);
targetsNotRecounted = num(indTargetsNotRecounted,:); 
num(indTargetsRecounted,:) = []; % make sure targets are removed from original data
num(indTargetsNotRecounted,:) = []; 
titles(indTargetsRecounted,:) = []; % make sure targets are removed from original data
titles(indTargetsNotRecounted,:) = []; 
raw(indTargetsRecounted,:) = []; % make sure targets are removed from original data
raw(indTargetsNotRecounted,:) = []; 

numNorm = mapminmax(num); % create a normalized version


% FINAL ROUND OF PREPROCESSING ----------------------------

parameters = {'a_rt90y', 'a_rt90x', 'dist/popPoints', 'distUppsala', ...
    'h_area', 'b_year', 'ARVARDE', 'r_sitearea', 'YearAroundOrVacation' ...
    'h_typbeby_fri_n', 'h_typbeby_kedj', 'h_typbeby_rad', 'h_typbeby_fri_YN', ...
    'r_assessmentyear', 'r_assessedvalue', 'r_recounted_assessedvalue', ...
    'r_stdsum', 'KODVA_full', 'KODVA_summer_w', 'KODVA_no', 'KODVA_B_YN', ...
    'KODVA_avl_YN', 'VGPredictions', 't_contractdate', 'yrs_since_contract', ...
    'summer_s', 'autumn_s', 'winter_s', 'spring_s', 'warm_cold_season'}; 

rL = horzcat(rL, parameters); 

% nParametersOrig = size(parameters);
% nColumnsOrig = length(num); 

trainFcns = {}; 
trainFcns{1, 1} = 'trainlm'; trainFcns{2, 1} = 'trainbr'; 

% possibleBorderDates = {'1-Oct-2016' '1-Nov-2016' '1-Dec-2016'}; 
possibleBorderDates = [736604 736635 736665]; 



% THE LOOP -----------------------------------------------------
% --------------------------------------------------------------
% --------------------------------------------------------------
outer_i = 2; 
keepGoing = true; 
while keepGoing
% for outer_i = 2:2
    
%     modded Inputs to be used in network
    
    
    % R1 RANDOMIZE PARAMETERS AND SET ANN OPTIONS -----------------------------------------

    neurons = layersNeurons(); 
    
    m_max_fail = floor(50 + rand * 500); 
    
    if neurons(1) > 60 || length(neurons) > 2 % to prevent 4ever effect
        m_max_fail = floor(50 + rand * 50);
    end
    
    % recounted/not recounted final prices: 
%     recountedPriceYN = floor(0.5 + rand); 
    recountedPriceYN = 1; 
    targets = targetsRecounted; 
    if recountedPriceYN == 0
        disp('not recounted'); 
        targets = targetsNotRecounted; 
    end
    
%     select a training function from list
    m_trainFcn = trainFcns{randi([1 numel(trainFcns)])}; 
    
    
  
%     R2 BUILD UP MATRIX USING SEMI-RANDOMIZED DATA -------------------------------------------
    
    newParameters = selectParameters(recountedPriceYN);
    
    input = zeros(length(newParameters), length(num)); 
    
    rL(outer_i, :) = {0}; 
    

%     store new parameters as binary results: 
    for i = startingIndexAfter:length(rL(1, :))
        indInRL = strcmp(newParameters(:, 1), rL(1,i));
        for j = 1:length(indInRL)
            if indInRL(j) == 1
                rL{outer_i, i} = 1; 
            end
        end
    end
    

%     append the data to new cell array which can be normalized
    
    normFlag = 0; 
    if rand < 0.5
        for i = 1:length(newParameters)
            indInNum = find(cellfun('length',regexp(titles,newParameters{i, 1})) == 1);
            input(i,:) = num(indInNum, :); 
        end
    else
        disp('normalizing'); 
        for i = 1:length(newParameters)
            indInNum = find(cellfun('length',regexp(titles,newParameters{i, 1})) == 1);
            input(i,:) = numNorm(indInNum, :); 
            normFlag = 1; 
        end
    end
    
    % Fix info on training/test/validation set
    % find index where training data stops
%     validation data from training data: HOLDOUT METHOD
    borderDate = datasample(possibleBorderDates, 1); 
    trainEnd = find(theDates > borderDate, 1, 'first');
    indexesTrain = 1:trainEnd; 
    indexesTest = trainEnd:length(theDates); 
    
    validationSetSize = floor(length(indexesTrain) / 6); 
    indexesValidation = randsample(indexesTrain, validationSetSize); 
%     remove from training set SHOULD BE CHECKED TO NOT HAVE DUPLICATES !!

    for i = 1:length(indexesValidation)
        index = find(indexesTrain == indexesValidation(i)); 
        if length(index) ~= 1
           throw(MException('something wrong when setting up validation set')); 
        end
        indexesTrain(index) = []; 
    end





%     INITIALIZE NETWORK ---------------------------------------------
% --------------------------------------------------------------
% --------------------------------------------------------------
    
    m_network = fitnet(neurons, m_trainFcn); 
    m_network.trainParam.max_fail = m_max_fail; 
    m_network.trainParam.epochs = 1000; 
    m_network.divideFcn = 'divideind'; 
    m_network.divideParam.trainInd = indexesTrain; 
    m_network.divideParam.valInd = indexesValidation; 
    m_network.divideParam.testInd = indexesTest; 
    
%     m_network.divideParam.trainRatio = 0.9; 
    

%     % simple ver
%     neurons = 6; 
%     m_network = fitnet(neurons); 
%     m_network.trainParam.max_fail = 5; 


%     RUN NETWORK -------------------------------------------

    disp(['run nr ' num2str(outer_i) ' with ' num2str(neurons) ...
        ' neurons, maxfail:' ...
        num2str(m_network.trainParam.max_fail) '--------------------']);
    
    
    m_cputime = cputime; 
    tic; 
    [trained_m_network, stats] = train(m_network, input, targets); 
    m_toc = toc; 
    m_cputime = cputime - m_cputime; 
    
    
    results = trained_m_network(input); 

    % MAPE MDAPE
    percentErrors = zeros(1, size(results, 2)); 

    for j = 1:size(results,2)
        percentErrors(1,j) = abs(1 - results(1,j) / targetsRecounted(1,j)); 
    end

    MAPE = mean(percentErrors); 
    MDAPE = median(percentErrors); 


    %  STORE RESULTS ---------------------------------------------
    rL{outer_i, 1} = MAPE;
    rL{outer_i, 2} = MDAPE; 
    rL{outer_i, 3} = mat2str(neurons); 
    rL{outer_i, 4} = '-';  
    rL{outer_i, 5} = stats.num_epochs; 
    rL{outer_i, 6} = stats.best_epoch; 
    rL{outer_i, 7} = m_network.trainFcn;
    rL{outer_i, 8} = stats.stop;
    rL{outer_i, 9} = m_cputime; 
    rL{outer_i, 10} = m_toc; 
    rL{outer_i, 11} = normFlag; 
    rL{outer_i, 12} = length(newParameters);
    rL{outer_i, 13} = datestr(borderDate); 

    
    dlmcell(name_output_file, rL); 
    
    outer_i = outer_i + 1; 
    
end







