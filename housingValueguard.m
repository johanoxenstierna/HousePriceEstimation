
clear; 
% clearvars -except sample inputs targets sheet_nr sheet_index name_output_file; 

% DO ANALYSIS ON SAVED DATA

rL = {}; 
rL{1, 1} = 'MAPE'; 
rL{1, 2} = 'MDAPE'; 
rL{1, 3} = 'neurons'; 
rL{1, 4} = 'epochs used'; 
rL{1, 5} = 'bestEpoch'; 
rL{1, 6} = 'trainFCN';
rL{1, 7} = 'reasonForStop';
rL{1, 8} = 'cpuTime';
rL{1, 9} = 'paramRemoved';

% OPEN OUTPUT FILE
prompt = 'What should be the name of the output file?  '; 
name_output_file = input(prompt); 

% if exist('name_output_file','var') == 0
%     name_output_file = 'output1'; 
%     
% else
%     fid = fopen(name_output_file,'a+');
% end


% LOAD SAMPLE if it doesnt already exist in workspace
% nParameters = 0; 
% if exist('sample','var') == 0
[samplen, titles] = xlsread('sample1.xlsx','lastFixes(mostClean)'); 
nParameters = size(samplen, 2) - 1;
targets = samplen(:, nParameters + 1)'; 
% inputs = samplen(:, 1:nParameters)'; 


% normalized data
inputs = mapminmax(samplen(:, 1:nParameters)'); 

stringParameter = ''; 

trainFcns = {}; 
trainFcns{1, 1} = 'trainlm'; 
trainFcns{2, 1} = 'trainbr'; 
trainFcns{3, 1} = 'trainbfg'; 
trainFcns{4, 1} = 'traincgb'; 
trainFcns{5, 1} = 'traincgf'; 
trainFcns{6, 1} = 'traincgp'; 
trainFcns{7, 1} = 'traingdm'; 
trainFcns{8, 1} = 'traingdx'; 
trainFcns{9, 1} = 'trainoss'; 
trainFcns{10, 1} = 'trainrp'; 
trainFcns{11, 1} = 'trainscg'; 
trainFcns{12, 1} = 'traingda'; 

% m_network.trainParam.min_grad = 0; 

% m_network.trainParam.trainFcn = 'trainlm'; 
% m_network.performFcn = 'mae'; 
% m_network.trainParam.lr = 0.5; 


fid = fopen(name_output_file,'a+');
fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ...
        rL{1, 1}, rL{1, 2}, rL{1, 3}, rL{1, 4}, rL{1, 5}, rL{1, 6}, rL{1, 7}, rL{1, 8}, rL{1, 9});

    
% THE LOOP -----------------------------------------------------
i = 2; 
keepGoing = true;
while keepGoing
    
%     modded Inputs to be used in network
    inputsMod = inputs; 
    stringParameter = ''; 
    % RANDOMIZATION -----------------------------------------
    
    m_trainFcn = char(trainFcns(floor(1 + rand * length(trainFcns)))); 
%     m_trainFcn = 'trainlm'; 
    number_of_neurons = floor(3 + rand() * 30); 
    m_max_fail = floor(50 + rand * 500); 
    % remove one parameter at random 
    removeParamYN = rand;
    if removeParamYN > 0.5
        disp('rem param'); 
        rrow = floor(1 + rand * nParameters); 
        inputsMod(rrow, :) = []; 
        stringParameter = char(titles(rrow)); 
       
    end

    % INITIALIZE NETWORK ---------------------------------------------
% 
    m_network = fitnet(number_of_neurons, m_trainFcn); 
    m_network.trainParam.max_fail = m_max_fail; 
    m_network.trainParam.epochs = 1000; 


%     % simple ver
%     m_network = fitnet(5, m_trainFcn); 
%     m_network.trainParam.max_fail = 30; 


%     RUN NETWORK -------------------------------------------

    disp(['run with ' num2str(number_of_neurons) ' neurons, maxfail:' num2str(m_network.trainParam.max_fail) '--------------------------------------------------']);
    
    m_cputime = cputime; 
    [trained_m_network, stats] = train(m_network, inputsMod, targets); 
    m_cputime = cputime - m_cputime; 
    
    results = trained_m_network(inputsMod); 
    perf = perform(trained_m_network, targets, results); 

    % MPE
    percentErrors = zeros(1, size(results, 2)); 

    for j = 1:size(results,2)
        percentErrors(1,j) = abs(1 - results(1,j) / targets(1,j)); 
    end

    MAPE = mean(percentErrors); 
    MDAPE = median(percentErrors); 


    %  STORE RESULTS ---------------------------------------------
    rL{i, 1} = MAPE;
    rL{i, 2} = MDAPE; 
    rL{i, 3} = number_of_neurons; 
    rL{i, 4} = stats.num_epochs; 
    rL{i, 5} = stats.best_epoch; 
    rL{i, 6} = m_network.trainFcn;
    rL{i, 7} = stats.stop;
    rL{i, 8} = m_cputime; 
    rL{i, 9} = stringParameter; 


    fprintf(fid, '%f\t%f\t%d\t%d\t%d\t%s\t%s\t%f\t%s\n', ...
        rL{i, 1}, rL{i, 2}, rL{i, 3}, rL{i, 4}, rL{i, 5}, rL{i, 6}, rL{i, 7}, rL{i, 8}, rL{i, 9});


%     disp(resultsLog([1 i],:))

%     sheet_index = strcat('A', num2str(i)); 

%     xlswrite('allResults.xlsx',resultsLog(i,:), sheet_nr, sheet_index);
%     dlmwrite('bigResult.csv', 'hjhj', '-append') ;


    i = i + 1; 
end

 






