function mat2json(dataDir)
% The function is part of PoseTrack dataset. 
%
% It converts labels from matlab structure loaded from a *mat file to pytohn dictionary and saves in a *json file
%
% dataDir (string): directory containing labels in *mat files
% Usage:
%       mat2json(dataDir)
%
% Example: 
%       mat2json('../../../posetrack_data/annotations/val/')
%

files = dir([dataDir '/*mat']);
fprintf('convert mat to json\n');
for i = 1:length(files)
    fprintf('%d/%d %s\n',i,length(files),files(i).name);
    filename = [dataDir '/' files(i).name];
    [p,n,~] = fileparts(filename);
    labels = load(filename);
    savejson('',labels,'FileName',[p '/' n '.json'],'SingletArray',1,'NaN','nan','Inf','inf');
end