%% Make sure you are in the right directory
cd(fileparts(matlab.desktop.editor.getActiveFilename));
if not (strcmp(version('-release'), '2024a'))
    warning("This script was tested on a different version of MATLAB.")
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%% Pipeline Options %%%%%%%%%%%%%%%%%%%%%%%%%%%%
save_values = 1;
show_plots = 0;

mav_kernel_width = 0.3; % Mean Absolute Value (MAV)
nfr_spike_threshold = -5; % Neural Firing Rate (NFR)
gs_correlation = .3; % Gram-Schmidt (GS)
gs_max_chans = 48;

selected_finger_movements = [1,2,3,4,5,6];
selected_participants = [1:4]; % Array of indices, where S5:(1), S6:(2), S7:(3), S8:(4)
selected_datasets = [0];  % Use 0 to use all data sets. Total number of datasets for each sheet = [102,164,155,71]

originalDir = 'C:\Users\Bret Mecham\Lab\Bret';
pipeline_directory = pwd;
output_directory = fullfile("..\..\results");
dataset_database = "C:\Users\Fredi Mino\Documents\research-projects\neural-features-embc-2025\data\database-01-dof.xlsx";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Pipeline
% Make Folders to save information if they do not exist
if ~exist(output_directory,'dir')
    error("Output directory not found")
end
%% 
tic
fprintf("Pipeline Started on %s\n", datetime)
for sheet_number = selected_participants
    % Initialize Table
    if sheet_number == 1
        participant = 'P2015';
    elseif sheet_number == 2
        participant = 'P2016';
    elseif sheet_number == 3
        participant = 'P2017';
    elseif sheet_number == 4
        participant = 'P2022';
    end
    fprintf('Now processing participant %s\n', participant)
    participant_db = readtable(dataset_database, 'Sheet', sheet_number, 'NumHeaderLines',0);
    if not(isequal(selected_datasets, [0]))
        participant_datasets = selected_datasets;
    else
        participant_datasets = [1:size(participant_db,1)];
    end
    for participant_dataset = participant_datasets
        if ~participant_db{participant_dataset, "ValidDataset"} % If not chosen, skip dataset
            fprintf("\tDataset %02d is invalid and has been skipped\n", participant_dataset)
            continue
        end
        % Get new files
        sections = split(participant_db{participant_dataset, "TrainingFilepaths"},'\');
        path_template = '\\%s\%s\sx\sx\s';
        temp = sprintf(path_template, sections{3}, sections{4}, sections{5}, sections{6});
        participant_dir_path  = sprintf('\\%s',temp);
        dataset_dir_path = sections{7};
        kdf_filename = [sections{end}];

        cd(fullfile(participant_dir_path, dataset_dir_path))

        ns5_file = dir('*.ns5').name;
        kdf_baseline_file = participant_db{participant_dataset,"BaselinesFilenames"};
        numChans = 192;
        cd(pipeline_directory)

        % Get New file info
        datasetPath = fullfile(participant_dir_path, dataset_dir_path);
        kdf_filepath = fullfile(dataset_dir_path, kdf_filename);
        kef_filepath = regexprep(fullfile(participant_dir_path, kdf_filepath),'.kdf','.kef');
        trial_struct = unrl_utils.parseKEF_jag(kef_filepath);

        [kinematics, Features, Targets, Kalman, nip_time_kdf] = unrl_utils.readKDF_jag(fullfile(participant_dir_path, kdf_filepath));

        % Extract information from files
        [neural_data_ns5, neural_baseline_ns5, nip_time_baseline] = load_ns5_baseline(strcat(datasetPath, '\'), ns5_file, kdf_baseline_file, nip_time_kdf);

        % Clear old dataset data
        clear AllFeatures Flexion Extension Both

        % Generate Neural Features
        nfr_features = makeNeuralFeatures_NS5(nfr_spike_threshold, nip_time_kdf, nip_time_baseline, neural_data_ns5, neural_baseline_ns5);
        mav_features = makeRollingPowerFeatures_zmh(nip_time_kdf, nip_time_baseline, neural_data_ns5, neural_baseline_ns5, mav_kernel_width);
        dwt_thresh = frm_wavedec.compute_dwt_thresholds(neural_baseline_ns5);
        dwt_features = frm_wavedec.compute_dwt_features(nip_time_kdf, neural_data_ns5, dwt_thresh)';% Transpose to match existing code

        % AllFeatures = [SpikeRates;MAVs; dwt_features];
        AllFeatures = [nfr_features; mav_features; dwt_features];

        % Find used Kinematics
        GoodUsedK = [];
        for j = selected_finger_movements
            if sum(find(kinematics(j,:)))
                GoodUsedK = [GoodUsedK, j];
            end
        end

        % Sperate training and testing sections for just combined movements
        [CombTrainMask, CombTestMask] = separateTrials(nfr_features,kinematics(GoodUsedK,:),trial_struct,nip_time_kdf,1,1,0.75);
        [TrainMask, TestMask] = separateTrials(nfr_features,kinematics(GoodUsedK,:),trial_struct,nip_time_kdf,0,0,0.75);
        CombTrainMask = logical(CombTrainMask - TrainMask);
        CombTestMask = logical(CombTestMask - TestMask);
        FlCombTrainMask = CombTrainMask;
        ExCombTrainMask = CombTrainMask;
        FlCombTestMask = CombTestMask;
        ExCombTestMask = CombTestMask;
    try
        % Isolate Flexions and Extensions in Training Mask
        [FlCombTrainMask,ExCombTrainMask,BothCombTrainMask] = IsolateMask(CombTrainMask,FlCombTrainMask,ExCombTrainMask,kinematics,GoodUsedK(1));
        % Isolate Flexions and Extensions in Testing Mask
        [FlCombTestMask,ExCombTestMask,BothCombTestMask] = IsolateMask(CombTestMask,FlCombTestMask,ExCombTestMask,kinematics,GoodUsedK(1));

        % Get Combined Flexion RMSEs
        if sum(FlCombTrainMask)
            [Flexion{13,1},Flexion{13,2},Flexion{13,3},Flexion{13,4},Flexion{13,5},Flexion{13,6},Flexion{13,7},Flexion{13,8},Flexion{13,9}, Flexion{13,10}] = Chans2RMSE(kinematics, nfr_features, mav_features, dwt_features, GoodUsedK(1), FlCombTrainMask, FlCombTestMask, gs_max_chans, gs_correlation);
        else
            for m = 1:9
                Flexion{13,m} = [];
            end
        end


        % Get Combined Extension RMSEs
        if sum(ExCombTrainMask)
            [Extension{13,1},Extension{13,2},Extension{13,3},Extension{13,4},Extension{13,5},Extension{13,6},Extension{13,7},Extension{13,8},Extension{13,9}, Extension{13,10}] = Chans2RMSE(kinematics, nfr_features, mav_features, dwt_features, GoodUsedK(1), ExCombTrainMask, ExCombTestMask, gs_max_chans, gs_correlation);
        else
            for m = 1:9
                Extension{13,m} = [];
            end
        end

        % Get Combined Flexing/Extension RMSEs
        if sum(FlCombTrainMask) && sum(ExCombTrainMask) % Must have Flexions and Extensions
            [Both{13,1},Both{13,2},Both{13,3},Both{13,4},Both{13,5},Both{13,6},Both{13,7},Both{13,8},Both{13,9}, Both{13,10}] = Chans2RMSE(kinematics, nfr_features, mav_features, dwt_features, GoodUsedK(1), BothCombTrainMask, BothCombTestMask, gs_max_chans, gs_correlation);
        else
            for m = 1:9
                Both{13,m} = [];
            end
        end
        close all
        for j = GoodUsedK
            % Sperate training and testing sections
            [TrainMask, TestMask] = separateTrials(nfr_features,kinematics(j,:),trial_struct,nip_time_kdf,0,0,0.75);
            % Initialize Flexion and Extension Masks
            FlTrainMask = TrainMask;
            ExTrainMask = TrainMask;
            FlTestMask = TestMask;
            ExTestMask = TestMask;

            % Isolate Flexions and Extensions in Training Mask
            [FlTrainMask,ExTrainMask,BothTrainMask] = IsolateMask(TrainMask,FlTrainMask,ExTrainMask,kinematics,j);
            % Isolate Flexions and Extensions in Testing Mask
            [FlTestMask,ExTestMask,BothTestMask] = IsolateMask(TestMask,FlTestMask,ExTestMask,kinematics,j);
            
            MAV1Masks{sheet_number}{participant_dataset}{j,1} = FlTrainMask;
            MAV1Masks{sheet_number}{participant_dataset}{j,2} = ExTrainMask;
            MAV1Masks{sheet_number}{participant_dataset}{j,3} = BothTrainMask;
            MAV1Masks{sheet_number}{participant_dataset}{j,4} = FlTestMask;
            MAV1Masks{sheet_number}{participant_dataset}{j,5} = ExTestMask;
            MAV1Masks{sheet_number}{participant_dataset}{j,6} = BothTestMask;
            % Get Flexion RMSEs
            if sum(FlTrainMask)
                [Flexion{j,1},Flexion{j,2},Flexion{j,3},Flexion{j,4},Flexion{j,5},Flexion{j,6},Flexion{j,7},Flexion{j,8},Flexion{j,9},Flexion{j,10}] = Chans2RMSE(kinematics, nfr_features, mav_features, dwt_features, j, FlTrainMask, FlTestMask, gs_max_chans, gs_correlation);
            else
                for m = 1:9
                    Flexion{j,m} = [];
                end
            end
            % Get Extension RMSEs
            if sum(ExTrainMask)
                [Extension{j,1},Extension{j,2},Extension{j,3},Extension{j,4},Extension{j,5},Extension{j,6},Extension{j,7},Extension{j,8},Extension{j,9},Extension{j,10}] = Chans2RMSE(kinematics, nfr_features, mav_features, dwt_features, j, ExTrainMask, ExTestMask, gs_max_chans, gs_correlation);
            else
                for m = 1:9
                    Extension{j,m} = [];
                end
            end
            % Get Flexing/Extension RMSEs
            if sum(FlTrainMask) && sum(ExTrainMask)  % Must have Flexions and Extensions
                [Both{j,1},Both{j,2},Both{j,3},Both{j,4},Both{j,5},Both{j,6},Both{j,7},Both{j,8},Both{j,9},Both{j,10}] = Chans2RMSE(kinematics, nfr_features, mav_features, dwt_features, j, BothTrainMask, BothTestMask, gs_max_chans, gs_correlation);
            else
                for m = 1:9
                    Both{j,m} = [];
                end
            end

            % Show Plots if Chosen
            if show_plots
                figure
                plot(Flexion{j,3})
                hold on
                plot(Flexion{j,4})
                hold on
                plot(kinematics(j,FlTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Flexion ', num2str(j)])

                figure
                plot(Extension{j,3})
                hold on
                plot(Extension{j,4})
                hold on
                plot(kinematics(j,ExTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Extension ', num2str(j)])

                figure
                plot(Both{j,3})
                hold on
                plot(Both{j,4})
                hold on
                plot(kinematics(j,BothTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Flexion and Extension ', num2str(j)])
            end
        end
        if show_plots
            if sum(FlCombTestMask)
                figure
                plot(Flexion{13,3})
                hold on
                plot(Flexion{13,4})
                hold on
                plot(kinematics(GoodUsedK(1),FlCombTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Combined Flexion ', num2str(13)])
            end

            if sum(ExCombTestMask)
                figure
                plot(Extension{13,3})
                hold on
                plot(Extension{13,4})
                hold on
                plot(kinematics(GoodUsedK(1),ExCombTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Combined Extension ', num2str(13)])
            end

            if sum(FlCombTrainMask) && sum(ExCombTrainMask)
                figure
                plot(Both{13,3})
                hold on
                plot(Both{13,4})
                hold on
                plot(kinematics(GoodUsedK(1),BothCombTestMask))
                legend('Spike','MAV','Kin')
                title(['Test Mask of Combined Flexion and Extension ', num2str(13)])
            end
        end

    % Save Data
    % MAV1Data: KDF name, AllFeatures, Channels Used, Flexion RMSEs, Extension RMSEs, Both RMSEs
    % Flexion/Extension/Both: SpikeChans, MAVChans, SpikeXhat, MAVXhat, SpikeRMSEs, MAVRMSEs
        MAV1Data{sheet_number}{participant_dataset,1} = kdf_filename;
        MAV1Data{sheet_number}{participant_dataset,2} = AllFeatures;
        if sum(FlCombTrainMask)
            MAV1Data{sheet_number}{participant_dataset,3} = [GoodUsedK 13];
        else
            MAV1Data{sheet_number}{participant_dataset,3} = GoodUsedK;
        end
        MAV1Data{sheet_number}{participant_dataset,4} = Flexion;
        MAV1Data{sheet_number}{participant_dataset,5} = Extension;
        MAV1Data{sheet_number}{participant_dataset,6} = Both;
        
        MAV1Masks{sheet_number}{participant_dataset}{7,1} = FlCombTrainMask;
        MAV1Masks{sheet_number}{participant_dataset}{7,2} = ExCombTrainMask;
        MAV1Masks{sheet_number}{participant_dataset}{7,3} = BothCombTrainMask;
        MAV1Masks{sheet_number}{participant_dataset}{7,4} = FlCombTestMask;
        MAV1Masks{sheet_number}{participant_dataset}{7,5} = ExCombTestMask;
        MAV1Masks{sheet_number}{participant_dataset}{7,6} = BothCombTestMask;
    catch
        MAV1Data{sheet_number}{participant_dataset,1} = "Error";
        disp('Error Occured')
    end
        if save_values
            % possibly change to save(filename,variables,'-append')
            save(fullfile('SavedInfo','MAV1Data'),'MAV1Data','-v7.3')
            save(fullfile('SavedInfo','MAV1Masks'),'MAV1Masks')
        end
        disp([participant, ', DataSet ' num2str(participant_dataset),' completed at: ',num2str(toc/60),' minutes'])
    end
end

%% Functions
% Loads the data from NS5 and Baseline Files. 
function [DNeural, NeuralBNS5, BNIPTime] = load_ns5_baseline(path, file, BKDFFile, KDFNIPTime)
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
    [HeaderNS5, DNS5] = unrl_utils.fastNSxRead2022('File',NS5File,'Range',Range);
    SfNS5 = (double(HeaderNS5.MaxAnlgVal(1))-double(HeaderNS5.MinAnlgVal(1)))/(double(HeaderNS5.MaxDigVal(1))-double(HeaderNS5.MinDigVal(1))); % scale factor for D2A conversion
    if size(DNS5,1) < 192
        numChans = size(DNS5,1);
    end
    DNeural = single((DNS5(1:numChans,:))').*SfNS5;
    
    % Read Baseline KDF
    disp('Reading Baseline Neural from NS5 file')
    [~,~,~,~,BNIPTime] = unrl_utils.readKDF_jag(fullfile(path, BKDFFile));
    BRange = [BNIPTime(1),BNIPTime(end)] + NIPOffset;
    [BHeader, BNS5] = unrl_utils.fastNSxRead2022('File',NS5File,'Range',BRange);
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