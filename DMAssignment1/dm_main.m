% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%               CSE 572: Data Mining - Assignment 1                       %
%                           Group 3                                       %
% Pratik Bartakke, Vihar Bhatt, Venkata Sai Sandeep Nadella, Darsh Parikh %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% Clearing all data from previous runs
clc
clear
close all

f = waitbar(0,'Please wait...');
pause(.5)

% Expects the dataset to be present at the root of current working directory.
% Note: The 'user9' folder under 'groundTruth' folder has been renamed to
% 'user09' to maintain consistency with 'user09' folder under 'MyoData'
% folder
% ./
% |-- Data_Mining/
% |   |-- groundTruth/
% |   |-- MyoData/
% |   |-- DataMining_project_help.docx
% |-- dm_main.m

addpath(genpath(pwd));  % Adding all the folders and subfolders of the current working directory to the work path

% Reading the sample dataset
waitbar(.33,f,'Loading the data');
pause(1)
fork_list = dir('Data_Mining/MyoData/*/fork/*');
fork_list = fork_list(~ismember({fork_list.name},{'.','..'}));
spoon_list = dir('Data_Mining/MyoData/*/spoon/*');
spoon_list = spoon_list(~ismember({spoon_list.name},{'.','..'}));
ground_truth_fork_list  = dir('Data_Mining/groundTruth/*/fork/*.txt');
ground_truth_spoon_list  = dir('Data_Mining/groundTruth/*/spoon/*.txt');
samples_matrix = [];    % Used for storing the entire sample data set after data cleaning

waitbar(.40,f,'Verifying the data');
pause(1)

if size(fork_list,1)~=(size(spoon_list,1))
    throw(MException('DM:DatasetMismatch', 'The number of fork and spoon eating action entries are not the same'));
end

if size(fork_list,1)/3~=size(ground_truth_fork_list,1)
    throw(MException('DM:DatasetMismatch', 'The number of eating actions and ground truth label entries are not the same'));
end

if size(ground_truth_fork_list,1)~=size(ground_truth_spoon_list,1)
    throw(MException('DM:DatasetMismatch', 'The number of ground truth labels for spoon and fork entries are not the same'));
end

num_users  = size(ground_truth_fork_list,1);

% The OOTB code only runs on fork dataset. Can be tweaked based on need to
% run on spoon dataset as well.
waitbar(.45,f,'Organizing data');
pause(1)
for i = 1:num_users
    
    path_split  = strsplit(ground_truth_fork_list(i).folder, '/');
    user  = path_split{end-1};  % contains the current user for which the data is being processed
    ground_truth_fork_data = csvread(ground_truth_fork_list(i).name);
    
    imu_fork_data = [];
    emg_fork_data = [];
    video_info_fork_data = [];
    
    %     Populate the dataset for current user
    for j=1:size(fork_list,1)
        if contains(fork_list(j).folder, user) && contains(fork_list(j).name,"IMU")
            imu_fork_data = csvread(fork_list(j).name);
        elseif contains(fork_list(j).folder, user) && contains(fork_list(j).name,"EMG")
            emg_fork_data = csvread(fork_list(j).name);
        elseif contains(fork_list(j).folder, user) && contains(fork_list(j).name,"video_info")
            video_info_fork_data = csvread(fork_list(j).name);
        end
        if isempty(imu_fork_data) || isempty(emg_fork_data) || isempty(video_info_fork_data)
            continue;
        else
            break;
        end
    end
    
    % Get the time axis of the video recording
    video_time_axis = linspace(max(imu_fork_data(1,1), emg_fork_data(1,1)),min(imu_fork_data(end,1), emg_fork_data(end,1)),(ground_truth_fork_data(end,2)-ground_truth_fork_data(1,1)+1));
    % Get the ground truth axis
    gt_time_axis = linspace(min(imu_fork_data(end,1), emg_fork_data(end,1))-(ground_truth_fork_data(end,2)*(1000/30)),min(imu_fork_data(end,1), emg_fork_data(end,1)),ground_truth_fork_data(end,2));
    
    gt_label = zeros(1,ground_truth_fork_data(end,2)); % the indices indicate ground truth frame axis
    for s=1:size(ground_truth_fork_data,1)
        for h = 1:length(gt_label)
            if h>=ground_truth_fork_data(s,1) && h<=ground_truth_fork_data(s,2)
                gt_label(1,h)=1; % Indicate it as eating frame
            end
        end
    end
    
    imu_fork_time = imu_fork_data(:,1);
    emg_fork_time = emg_fork_data(:,1);
    
    %     interpolate the IMU fork data to a common video time axis to combine
    %     IMU, EMG and GroundTruth data
    interpolated_fork_data = [];
    interpolated_imu_data = [];
    interpolated_emg_data = [];
    interpolated_gt_data = [];
    interpolated_fork_data_cleaned = [];
    
    for k = 2:11    % there are 18 sensors in the given dataset
        interpolated_imu_data = vertcat(interpolated_imu_data,interp1(imu_fork_time, imu_fork_data(:,k), video_time_axis, 'linear'));
        if k~=10 && k~=11
            interpolated_emg_data = vertcat(interpolated_emg_data,interp1(emg_fork_time, emg_fork_data(:,k), video_time_axis, 'linear'));
        end
    end
    interpolated_gt_data = interp1(gt_time_axis, gt_label, video_time_axis, 'linear');
    interpolated_fork_data = vertcat(interpolated_gt_data, interpolated_imu_data, interpolated_emg_data);
    interpolated_fork_data = vertcat(repmat(str2num(strrep(user,'user','')),1,length(video_time_axis)), video_time_axis, interpolated_fork_data);
    
    for g = 1:size(interpolated_fork_data, 2)
        if (interpolated_fork_data(3,g)==1||interpolated_fork_data(3,g)==0)
            interpolated_fork_data_cleaned = [interpolated_fork_data_cleaned interpolated_fork_data(:,g)];
        end
    end
    
    % concatenate and store the data corresponding to each user along
    % with timestamps across all users in the samples_matrix
    samples_matrix = [samples_matrix interpolated_fork_data_cleaned];
    
end

clearvars -except f samples_matrix


feature_eat_matrix = [];
feature_noneat_matrix = [];

prev = 0;
curr=0;
t_mat = [];

% Compute the feature matrix. 6 features were considered for this
% assignment. 'mean', 'max', 'rms', 'std', 'entropy', 'power'
waitbar(.50,f,'Extracting features');
pause(1)
for i = 1:size(samples_matrix,2)
    curr = samples_matrix(3,i);
    if (curr == 0 && prev == 0) || (curr==1 && prev==1)
        t_mat = [t_mat samples_matrix(4:end,i)];
    elseif curr==0 && prev==1
        fft_feature = abs(fft(t_mat,[],2));
        power = fft_feature.*conj(fft_feature)/size(t_mat,2);
        total_power = sum(power,2);
        
        p=power;
        p=p/sum(p+ 1e-12);
        logd = log2(p + 1e-12);
        entropy = -sum(p.*logd,2)/log2(length(p));
        
        t_mat = vertcat(mean(t_mat,2), max(t_mat,[],2), rms(t_mat,2), std(t_mat,[],2), entropy, total_power);
        feature_eat_matrix = [feature_eat_matrix t_mat];
        t_mat=samples_matrix(4:end,i);
    else
        fft_feature = abs(fft(t_mat,[],2));
        power = fft_feature.*conj(fft_feature)/size(t_mat,2);
        total_power = sum(power,2);
        
        p=power;
        p=p/sum(p+ 1e-12);
        logd = log2(p + 1e-12);
        entropy = -sum(p.*logd,2)/log2(length(p));
        
        t_mat = vertcat(mean(t_mat,2), max(t_mat,[],2), rms(t_mat,2), std(t_mat,[],2), entropy, total_power);
        feature_noneat_matrix = [feature_noneat_matrix t_mat];
        t_mat=samples_matrix(4:end,i);
    end
    prev=curr;
end
row_labels = {'OrientationX mean','OrientationY mean','OrientationZ mean','OrientationW mean','AccelerometerX mean','AccelerometerY mean','AccelerometerZ mean','GyroscopeX mean','GyroscopeY mean','GyroscopeZ mean','EMG1 mean','EMG2 mean','EMG3 mean','EMG4 mean','EMG5 mean','EMG6 mean','EMG7 mean','EMG8 mean','OrientationX max','OrientationY max','OrientationZ max','OrientationW max','AccelerometerX max','AccelerometerY max','AccelerometerZ max','GyroscopeX max','GyroscopeY max','GyroscopeZ max','EMG1 max','EMG2 max','EMG3 max','EMG4 max','EMG5 max','EMG6 max','EMG7 max','EMG8 max','OrientationX rms','OrientationY rms','OrientationZ rms','OrientationW rms','AccelerometerX rms','AccelerometerY rms','AccelerometerZ rms','GyroscopeX rms','GyroscopeY rms','GyroscopeZ rms','EMG1 rms','EMG2 rms','EMG3 rms','EMG4 rms','EMG5 rms','EMG6 rms','EMG7 rms','EMG8 rms','OrientationX std','OrientationY std','OrientationZ std','OrientationW std','AccelerometerX std','AccelerometerY std','AccelerometerZ std','GyroscopeX std','GyroscopeY std','GyroscopeZ std','EMG1 std','EMG2 std','EMG3 std','EMG4 std','EMG5 std','EMG6 std','EMG7 std','EMG8 std','OrientationX entropy','OrientationY entropy','OrientationZ entropy','OrientationW entropy','AccelerometerX entropy','AccelerometerY entropy','AccelerometerZ entropy','GyroscopeX entropy','GyroscopeY entropy','GyroscopeZ entropy','EMG1 entropy','EMG2 entropy','EMG3 entropy','EMG4 entropy','EMG5 entropy','EMG6 entropy','EMG7 entropy','EMG8 entropy','OrientationX power','OrientationY power','OrientationZ power','OrientationW power','AccelerometerX power','AccelerometerY power','AccelerometerZ power','GyroscopeX power','GyroscopeY power','GyroscopeZ power','EMG1 power','EMG2 power','EMG3 power','EMG4 power','EMG5 power','EMG6 power','EMG7 power','EMG8 power'};
samples_labels = ['user', 'time_stamp', 'ground_truth', "OrientationX","OrientationY","OrientationZ","OrientationW","AccelerometerX","AccelerometerY","AccelerometerZ","GyroscopeX","GyroscopeY","GyroscopeZ","EMG1","EMG2","EMG3","EMG4","EMG5","EMG6","EMG7","EMG8"];

% Refer to the below tables for getting a visual sense of the samples, feature
% matrices. Warning: Doesn't contain all observations. Just for visualization
% purpose.
feature_eat_matrix_table  = array2table(feature_eat_matrix,'RowNames', row_labels);
feature_noneat_matrix_table  = array2table(feature_noneat_matrix,'RowNames', row_labels);
samples_table = array2table(samples_matrix(:,1:2000),'RowNames',samples_labels);

% Compute the cosine similarity between the plots to get the
% dissimilar features with more distance
waitbar(.73,f,'Computing cosine distance');
pause(1)
cosine_similarity=[];
feature_noneat_mat=feature_noneat_matrix(:,(1:1160)); % Discard one observation and compare the equal number of eating and non eating observations
num_features = size(feature_eat_matrix_table,1);
for k=1:num_features
    cosine_similarity=[cosine_similarity 1 - pdist([feature_eat_matrix(k,:); feature_noneat_mat(k,:)], 'cosine')];
end
mean_cosine_sim = mean(cosine_similarity,2);

picked_features=[];

% Pick the features with cosine similarity less(or highly dissimilar/distant) than the mean cosine
% similarity across all the observations
for k=1:num_features
    m = 1 - pdist([feature_eat_matrix(k,:); feature_noneat_mat(k,:)], 'cosine');
    if m>=0 && m<mean_cosine_sim
        t = [feature_eat_matrix(k,:) feature_noneat_matrix(k,:)];
        picked_features = vertcat(picked_features, t);
        figure('Name',row_labels{k});
        clf
        plot(1:size(feature_eat_matrix,2),smooth(feature_eat_matrix(k,:)));
        hold on;
        plot(1:size(feature_noneat_matrix,2),smooth(feature_noneat_matrix(k,:)));
        xlabel('Number of samples');
        ylabel(row_labels{k});
        legend('Eating','Noneating');
        title([row_labels{k} ' Eating vs Noneating']);
    end
end

% Apply PCA over the distant features picked. Refer '>> doc pca' for more info
% how PCA works and the outputs returned
waitbar(.80,f,'Running PCA');
pause(1)
[coeff,score,latent,tsquared,explained,mu]=pca(picked_features.');
% Observe that the top 11 variances constitute to 100% variance and hence
% we can safely pick only these components
explained

% Plotting the first 3 Principal Components
figure('Name','Biplot for PC1, PC2, PC3');
biplot(coeff(:,1:3),'scores',score(:,1:3));
title('Biplot for PC1, PC2, PC3');

legend_eigen={};
figure('Name','Plot for Eigen vectors');
for i=1:11
    plot(coeff(:,i));
    hold on;
    xlabel('Number of samples');
    ylabel('Eigen Vectors for Principal Components');
    legend_eigen{i} = ['Eigen Vector ' num2str(i)];
end
title('Plot for Eigen vectors');
legend(legend_eigen);

new_feature_mat = score;
new_feature_mat = [new_feature_mat vertcat(ones(1160,1), zeros(1161,1))];

for i=1:11
    figure('Name',['Plot of new feature matrix for PC ' num2str(i) ' with variance ' num2str(explained(i,1)) '%'])
    plot(smooth(new_feature_mat([1:1160],i)));
    hold on;
    plot(smooth(new_feature_mat([1161:2321],i)));
    xlabel('Number of samples');
    ylabel('Principle Component Values');
    title(['Plot of new feature matrix for PC ' num2str(i) ' with variance ' num2str(explained(i,1)) '%']);
    legend('Eating','Noneating');
end

waitbar(.90,f,'Saving plots');
pause(1)

figures_folder_name = 'figures'; % Destination folder for storing the plots
mkdir(figures_folder_name);
figures_list = findobj(allchild(0), 'flat', 'Type', 'figure');
for i = 1:length(figures_list)
    figure_handle = figures_list(i);
    figure_name   = get(figure_handle, 'Name');
    if figure_name~=""
        savefig(figure_handle, fullfile(figures_folder_name, [figure_name, '.fig']));
        saveas(figure_handle, fullfile(figures_folder_name, [figure_name, '.png']),'png');
    end
end

waitbar(1,f,'Finishing');
pause(1)
close(f)
clearvars -except samples_table feature_eat_matrix_table feature_noneat_matrix_table
