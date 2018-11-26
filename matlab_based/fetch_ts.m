% Fetch raw csv data from Fred. as time sereis.

% function ts = fetch_ts(config)
load("config.mat")
addpath("data")
%% Get data from source file
data = readtable(config.fileName, "Delimiter", ",", "ReadVariableNames", true);

timeVals = datestr(datenum(data.DATE), "dd-mmm-yyyy");
dataVals = double(data.UNRATE);

ts = timeseries(dataVals, timeVals, "Name", "UNRATE monthly");
ts.TimeInfo.Format = "mmm yyyy"; % change format of datetime on plot

% end