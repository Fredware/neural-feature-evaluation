%% See if MAV can train on 1 DoF (fist)

%%%%%%%%%%%%%%%%%%%CHANGES%%%%%%%%%%%%%%%%%%%
saveValues = 1;
ShowPlots = 0;
KernelWidth = 0.3; % MAV
Threshold = -5; % Spike
cor = .3; % GS
FingerMovements = [1,2,3,4,5,6];
maxChans = 48;
UsedParticipants = [1:4]; % S5(1) - S6(2) - S7(3) - S8(4)
originalDir = 'C:\Users\Bret Mecham\Lab\Bret';
Datasets = [0];  % Use 0 to use all data sets, Total number of datasets for each sheet [102,164,155,71] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make Folders to save information if they do not exist
if ~exist(fullfile('SavedInfo'),'dir')
    mkdir SavedInfo
end

tic
for SheetNumber = UsedParticipants
    % Initialize Table
    if SheetNumber == 1
        participant = 'P2015';
    elseif SheetNumber == 2
        participant = 'P2016';
    elseif SheetNumber == 3
        participant = 'P2017';
    elseif SheetNumber == 4
        participant = 'P2022';
    end
    disp(['Now on ', participant])
    T = readtable('C:\Users\Bret Mecham\Lab\Bret\MAVvsSpikeDOF1.xlsx','Sheet',SheetNumber,'NumHeaderLines',0);
    if Datasets
        sets = Datasets;
    else
        sets = [1:size(T,1)];
    end
    for i = sets
        if ~T{i,10} % If not chosen, skip dataset
            disp([num2str(i),': Skipped'])
            continue
        end
        % Get new files
        sections = split(T{i,2},'\');
        makePath = '\\%s\%s\sx\sx\s';
        temp = sprintf(makePath,sections{3},sections{4},sections{5},sections{6});
        finalPath = sprintf('\\%s',temp);

        path = finalPath;
        DatasetFiles = sections{7};
        KDFFiles = [sections{end}];

        cd(fullfile(finalPath,sections{7}))

        NS5Files = dir('*.ns5').name;
        BaselineFiles = T{i,3};
        numChans = 192;
        cd(originalDir)

        % Get New file info
        datasetPath = fullfile(path, DatasetFiles);
        KDFFile = fullfile(DatasetFiles, KDFFiles);
        KEFFile = regexprep(fullfile(path, KDFFile),'.kdf','.kef');
        TrialStruct = parseKEF_jag(KEFFile);

        [Kinematics,Features,Targets,Kalman,KDFNIPTime] = readKDF(fullfile(path, KDFFile));

        file = NS5Files;
        BKDFFile = BaselineFiles;

        % Extract information from files
        [DNeural, NeuralBNS5, BNIPTime] = load_NSF_Baseline(strcat(datasetPath, '\'), file, BKDFFile, KDFNIPTime);

        % Clear old dataset data
        clear AllFeatures Flexion Extension Both

        % Generate Neural Features
        SpikeRates = makeNeuralFeatures_NS5(Threshold, KDFNIPTime, BNIPTime, DNeural, NeuralBNS5);
        MAVs = makeRollingPowerFeatures_zmh(KDFNIPTime, BNIPTime, DNeural, NeuralBNS5, KernelWidth);
        dwt_thresh = frm_wavedec.compute_dwt_thresholds(NeuralBNS5);
        dwt_features = frm_wavedec.compute_dwt_features(KDFNIPTime, DNeural, dwt_thresh)';% Transpose to match existing code

        % AllFeatures = [SpikeRates;MAVs; dwt_features];
        AllFeatures = [SpikeRates; MAVs; dwt_features];

        % Find used Kinematics
        GoodUsedK = [];
        for j = FingerMovements
            if sum(find(Kinematics(j,:)))
                GoodUsedK = [GoodUsedK, j];
            end
        end

        % Sperate training and testing sections for just combined movements
        [CombTrainMask, CombTestMask] = separateTrials(SpikeRates,Kinematics(GoodUsedK,:),TrialStruct,KDFNIPTime,1,1,0.75);
        [TrainMask, TestMask] = separateTrials(SpikeRates,Kinematics(GoodUsedK,:),TrialStruct,KDFNIPTime,0,0,0.75);
        CombTrainMask = logical(CombTrainMask - TrainMask);
        CombTestMask = logical(CombTestMask - TestMask);
        FlCombTrainMask = CombTrainMask;
        ExCombTrainMask = CombTrainMask;
        FlCombTestMask = CombTestMask;
        ExCombTestMask = CombTestMask;
    try
        % Isolate Flexions and Extensions in Training Mask
        [FlCombTrainMask,ExCombTrainMask,BothCombTrainMask] = IsolateMask(CombTrainMask,FlCombTrainMask,ExCombTrainMask,Kinematics,GoodUsedK(1));
        % Isolate Flexions and Extensions in Testing Mask
        [FlCombTestMask,ExCombTestMask,BothCombTestMask] = IsolateMask(CombTestMask,FlCombTestMask,ExCombTestMask,Kinematics,GoodUsedK(1));

        % Get Combined Flexion RMSEs
        if sum(FlCombTrainMask)
            [Flexion{13,1},Flexion{13,2},Flexion{13,3},Flexion{13,4},Flexion{13,5},Flexion{13,6},Flexion{13,7},Flexion{13,8},Flexion{13,9}, Flexion{13,10}] = Chans2RMSE(Kinematics, SpikeRates, MAVs, dwt_features, GoodUsedK(1), FlCombTrainMask, FlCombTestMask, maxChans, cor);
        else
            for m = 1:9
                Flexion{13,m} = [];
            end
        end


        % Get Combined Extension RMSEs
        if sum(ExCombTrainMask)
            [Extension{13,1},Extension{13,2},Extension{13,3},Extension{13,4},Extension{13,5},Extension{13,6},Extension{13,7},Extension{13,8},Extension{13,9}, Extension{13,10}] = Chans2RMSE(Kinematics, SpikeRates, MAVs, dwt_features, GoodUsedK(1), ExCombTrainMask, ExCombTestMask, maxChans, cor);
        else
            for m = 1:9
                Extension{13,m} = [];
            end
        end

        % Get Combined Flexing/Extension RMSEs
        if sum(FlCombTrainMask) && sum(ExCombTrainMask) % Must have Flexions and Extensions
            [Both{13,1},Both{13,2},Both{13,3},Both{13,4},Both{13,5},Both{13,6},Both{13,7},Both{13,8},Both{13,9}, Both{13,10}] = Chans2RMSE(Kinematics, SpikeRates, MAVs, dwt_features, GoodUsedK(1), BothCombTrainMask, BothCombTestMask, maxChans, cor);
        else
            for m = 1:9
                Both{13,m} = [];
            end
        end
        close all
        for j = GoodUsedK
            % Sperate training and testing sections
            [TrainMask, TestMask] = separateTrials(SpikeRates,Kinematics(j,:),TrialStruct,KDFNIPTime,0,0,0.75);
            % Initialize Flexion and Extension Masks
            FlTrainMask = TrainMask;
            ExTrainMask = TrainMask;
            FlTestMask = TestMask;
            ExTestMask = TestMask;

            % Isolate Flexions and Extensions in Training Mask
            [FlTrainMask,ExTrainMask,BothTrainMask] = IsolateMask(TrainMask,FlTrainMask,ExTrainMask,Kinematics,j);
            % Isolate Flexions and Extensions in Testing Mask
            [FlTestMask,ExTestMask,BothTestMask] = IsolateMask(TestMask,FlTestMask,ExTestMask,Kinematics,j);
            
            MAV1Masks{SheetNumber}{i}{j,1} = FlTrainMask;
            MAV1Masks{SheetNumber}{i}{j,2} = ExTrainMask;
            MAV1Masks{SheetNumber}{i}{j,3} = BothTrainMask;
            MAV1Masks{SheetNumber}{i}{j,4} = FlTestMask;
            MAV1Masks{SheetNumber}{i}{j,5} = ExTestMask;
            MAV1Masks{SheetNumber}{i}{j,6} = BothTestMask;
            % Get Flexion RMSEs
            if sum(FlTrainMask)
                [Flexion{j,1},Flexion{j,2},Flexion{j,3},Flexion{j,4},Flexion{j,5},Flexion{j,6},Flexion{j,7},Flexion{j,8},Flexion{j,9},Flexion{j,10}] = Chans2RMSE(Kinematics, SpikeRates, MAVs, dwt_features, j, FlTrainMask, FlTestMask, maxChans, cor);
            else
                for m = 1:9
                    Flexion{j,m} = [];
                end
            end
            % Get Extension RMSEs
            if sum(ExTrainMask)
                [Extension{j,1},Extension{j,2},Extension{j,3},Extension{j,4},Extension{j,5},Extension{j,6},Extension{j,7},Extension{j,8},Extension{j,9},Extension{j,10}] = Chans2RMSE(Kinematics, SpikeRates, MAVs, dwt_features, j, ExTrainMask, ExTestMask, maxChans, cor);
            else
                for m = 1:9
                    Extension{j,m} = [];
                end
            end
            % Get Flexing/Extension RMSEs
            if sum(FlTrainMask) && sum(ExTrainMask)  % Must have Flexions and Extensions
                [Both{j,1},Both{j,2},Both{j,3},Both{j,4},Both{j,5},Both{j,6},Both{j,7},Both{j,8},Both{j,9},Both{j,10}] = Chans2RMSE(Kinematics, SpikeRates, MAVs, dwt_features, j, BothTrainMask, BothTestMask, maxChans, cor);
            else
                for m = 1:9
                    Both{j,m} = [];
                end
            end

            % Show Plots if Chosen
            if ShowPlots
                figure
                plot(Flexion{j,3})
                hold on
                plot(Flexion{j,4})
                hold on
                plot(Kinematics(j,FlTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Flexion ', num2str(j)])

                figure
                plot(Extension{j,3})
                hold on
                plot(Extension{j,4})
                hold on
                plot(Kinematics(j,ExTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Extension ', num2str(j)])

                figure
                plot(Both{j,3})
                hold on
                plot(Both{j,4})
                hold on
                plot(Kinematics(j,BothTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Flexion and Extension ', num2str(j)])
            end
        end
        if ShowPlots
            if sum(FlCombTestMask)
                figure
                plot(Flexion{13,3})
                hold on
                plot(Flexion{13,4})
                hold on
                plot(Kinematics(GoodUsedK(1),FlCombTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Combined Flexion ', num2str(13)])
            end

            if sum(ExCombTestMask)
                figure
                plot(Extension{13,3})
                hold on
                plot(Extension{13,4})
                hold on
                plot(Kinematics(GoodUsedK(1),ExCombTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Combined Extension ', num2str(13)])
            end

            if sum(FlCombTrainMask) && sum(ExCombTrainMask)
                figure
                plot(Both{13,3})
                hold on
                plot(Both{13,4})
                hold on
                plot(Kinematics(GoodUsedK(1),BothCombTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Combined Flexion and Extension ', num2str(13)])
            end
        end

    % Save Data
    % MAV1Data: KDF name, AllFeatures, Channels Used, Flexion RMSEs, Extension RMSEs, Both RMSEs
    % Flexion/Extension/Both: SpikeChans, MAVChans, SpikeXhat, MAVXhat, SpikeRMSEs, MAVRMSEs
        MAV1Data{SheetNumber}{i,1} = KDFFiles;
        MAV1Data{SheetNumber}{i,2} = AllFeatures;
        if sum(FlCombTrainMask)
            MAV1Data{SheetNumber}{i,3} = [GoodUsedK 13];
        else
            MAV1Data{SheetNumber}{i,3} = GoodUsedK;
        end
        MAV1Data{SheetNumber}{i,4} = Flexion;
        MAV1Data{SheetNumber}{i,5} = Extension;
        MAV1Data{SheetNumber}{i,6} = Both;
        
        MAV1Masks{SheetNumber}{i}{7,1} = FlCombTrainMask;
        MAV1Masks{SheetNumber}{i}{7,2} = ExCombTrainMask;
        MAV1Masks{SheetNumber}{i}{7,3} = BothCombTrainMask;
        MAV1Masks{SheetNumber}{i}{7,4} = FlCombTestMask;
        MAV1Masks{SheetNumber}{i}{7,5} = ExCombTestMask;
        MAV1Masks{SheetNumber}{i}{7,6} = BothCombTestMask;
    catch
        MAV1Data{SheetNumber}{i,1} = "Error";
        disp('Error Occured')
    end
        if saveValues
            % possibly change to save(filename,variables,'-append')
            save(fullfile('SavedInfo','MAV1Data'),'MAV1Data','-v7.3')
            save(fullfile('SavedInfo','MAV1Masks'),'MAV1Masks')
        end
        disp([participant, ', DataSet ' num2str(i),' completed at: ',num2str(toc/60),' minutes'])
    end
end

%% Functions
% Loads the data from NS5 and Baseline Files. 
function [DNeural, NeuralBNS5, BNIPTime] = load_NSF_Baseline(path, file, BKDFFile, KDFNIPTime)
    numChans = 192;
    path = char(path);
    file = char(file);
    BKDFFile = char(BKDFFile);
    
    % Read NS5
    NS5File = fullfile(path,file); 
    disp(NS5File);
    NS2File = regexprep(NS5File,'.ns5','.ns2');
    RecStartFile = fullfile(path, ['RecStart_', path(end-15:end-1), '.mat']);
    try
        NIPOffset = CalculateNIPOffset(NS2File, RecStartFile);
    catch
        RecStartFile = fullfile(path, ['Kalman_SSStruct_', path(end-15:end-1), '.mat']); % P2015 has different RecStart.mat file
        NIPOffset = CalculateNIPOffset(NS2File, RecStartFile);
    end
    Range = [KDFNIPTime(1),KDFNIPTime(end)] + NIPOffset;  %%% NIP Offset is the number of NS2 samples leading the KDF
    disp('Reading Neural Data from NS5 file')
    [HeaderNS5, DNS5] = fastNSxRead2022('File',NS5File,'Range',Range);
    SfNS5 = (double(HeaderNS5.MaxAnlgVal(1))-double(HeaderNS5.MinAnlgVal(1)))/(double(HeaderNS5.MaxDigVal(1))-double(HeaderNS5.MinDigVal(1))); % scale factor for D2A conversion
    if size(DNS5,1) < 192
        numChans = size(DNS5,1);
    end
    DNeural = single((DNS5(1:numChans,:))').*SfNS5;
    
    % Read Baseline KDF
    disp('Reading Baseline Neural from NS5 file')
    [~,~,~,~,BNIPTime] = readKDF(fullfile(path, BKDFFile));
    BRange = [BNIPTime(1),BNIPTime(end)] + NIPOffset;
    [BHeader, BNS5] = fastNSxRead2022('File',NS5File,'Range',BRange);
    SfBNS5 = (double(BHeader.MaxAnlgVal(1))-double(BHeader.MinAnlgVal(1)))/(double(BHeader.MaxDigVal(1))-double(BHeader.MinDigVal(1))); % scale factor for dig2analog
    NeuralBNS5 = single(BNS5(1:numChans,:)')*SfBNS5;
end

function [SpikeChans,MAVChans,AllChans,SpikeXhat,MAVXhat,AllXhat,SpikeRMSEs,MAVRMSEs, DWTRMSEs, AllRMSEs] = Chans2RMSE(Kinematics, SpikeRates, MAVs, dwt_features, Kin, TrainMask, TestMask, maxChans, minCorrelation)
    % Select Channels
    AllFeats = [SpikeRates;MAVs; dwt_features];
    SChans = gramSchmDarpa_jag(Kinematics(Kin,TrainMask),SpikeRates(:,TrainMask),1,maxChans,0,'none','all',minCorrelation); %.2, 0, none
    MChans = gramSchmDarpa_jag(Kinematics(Kin,TrainMask),MAVs(:,TrainMask),1,maxChans,0,'none','all',minCorrelation); %.2, 0, none
    DChans = gramSchmDarpa_jag(Kinematics(Kin,TrainMask),dwt_features(:,TrainMask),1,maxChans,0,'none','all',minCorrelation); %.2, 0, none
    AChans = gramSchmDarpa_jag(Kinematics(Kin,TrainMask),AllFeats(:,TrainMask),1,maxChans,0,'none','all',minCorrelation); %.2, 0, none
    SpikeChans = SChans{1};
    MAVChans = MChans{1};
    DWTChans = DChans{1};
    AllChans = AChans{1};

    % Train MKF and test inferences
    SpikeTRAIN = trainDecode_jag(Kinematics(Kin,TrainMask), SpikeRates(:,TrainMask), SpikeChans, 'standard');
    [SpikeXhat] = runDecode_jag(SpikeTRAIN, Kinematics(Kin,TestMask), SpikeRates(:,TestMask), SpikeChans, 'standard');
    
    MAVTRAIN = trainDecode_jag(Kinematics(Kin,TrainMask), MAVs(:,TrainMask), MAVChans, 'standard');
    [MAVXhat] = runDecode_jag(MAVTRAIN, Kinematics(Kin,TestMask), MAVs(:,TestMask), MAVChans, 'standard');

    DWTTRAIN = trainDecode_jag(Kinematics(Kin,TrainMask), dwt_features(:,TrainMask), DWTChans, 'standard');
    [DWTXhat] = runDecode_jag(DWTTRAIN, Kinematics(Kin,TestMask), dwt_features(:,TestMask), DWTChans, 'standard');

    AllTRAIN = trainDecode_jag(Kinematics(Kin,TrainMask), AllFeats(:,TrainMask), AllChans, 'standard');
    [AllXhat] = runDecode_jag(AllTRAIN, Kinematics(Kin,TestMask), AllFeats(:,TestMask), AllChans, 'standard');

    % Get RMSEs
    [SpikeRMSEs] = getRMSE(Kinematics(Kin,TestMask),SpikeXhat);
    [MAVRMSEs] = getRMSE(Kinematics(Kin,TestMask),MAVXhat);
    [DWTRMSEs] = getRMSE(Kinematics(Kin,TestMask),DWTXhat);
    [AllRMSEs] = getRMSE(Kinematics(Kin,TestMask),AllXhat);
end
    
function [FlMask, ExMask, Mask] = IsolateMask(Mask,FlMask,ExMask,Kinematics,ChosenKins)
    % Find Train Windows
    StartStops = diff(Mask);
    StartInds = find(StartStops == 1);
    StopInds = find(StartStops == -1);
    if(length(StopInds) < length(StartInds))
        StopInds = [StopInds length(Mask)];
    end

    if(length(StartInds) ~= length(StopInds))
        StartInds = [1 StartInds];
    end

    % Remove Train windows of unused kinematics
    for k = 1:length(StartInds)
        wind = [StartInds(k):StopInds(k)];
        if sum(Kinematics(ChosenKins,wind)) <= 0
            FlMask(wind) = zeros(1,numel(wind));   % Get rid of 0 and Extensions for Flexion Train Mask
        end
        if sum(Kinematics(ChosenKins,wind)) >= 0
            ExMask(wind) = zeros(1,numel(wind));   % Get rid of 0 and Flexions for Extension Train Mask
        end
        if sum(Kinematics(ChosenKins,wind)) == 0
            Mask(wind) = zeros(1,numel(wind));   % Get rid of 0 and Extensions for Flexion Train Mask
        end
    end
%         figure
%     plot(Kinematics(ChosenKins,:))
%     hold on
%     plot(FlMask)
%         figure
%     plot(Kinematics(ChosenKins,:))
%     hold on
%     plot(ExMask)
%         figure
%     plot(Kinematics(ChosenKins,:))
%     hold on
%     plot(Mask)
end