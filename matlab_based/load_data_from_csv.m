% Collect time timetable values from csv.

addpath('data');

% Name of variable to be loaded into workspace.
names = {
	'UNRATE',  % (Target) Civilian Unemployment Rate
	'INDPRO',  % Industrial Production Index
	'M2',  % M2 Money Stock
	'CPIAUCSL',  % Consumer Price Index for All Urban Consumers: All Items
% 	'SP500',  % S&P 500
% 	'AAA',  % Moody's Seasoned Aaa Corporate Bond Yield
% 	'PSAVERT',  % Personal Saving Rate
%  	'GDPC1', % Real Gross Domestic Product
	};

tt_stack = {};

% Load data to workspace.
for i = 1: length(names)
	table = readtable([names{i}, '.csv']);
	tt = table2timetable(table);
	tt_stack{i} = retime(tt, 'monthly', 'linear');
end

%%
data.tt = synchronize(tt_stack{:});
% Special Consideration.
% data.tt.SP500 = str2double(data.tt.SP500);

% data.tt = retime(data.tt, 'monthly', 'linear');

%%
data.tt = rmmissing(data.tt); %Remove missing variables.

% Taking data range.
% data.tt = data.tt(timerange('1998-06-01', '2018-02-01'), :);

data.tb = timetable2table(data.tt);
% data.ar = table2array(data.tt(:, 2:end))'; % Collection of numerical data
data.ar = data.tt.Variables;
% with DATE variable removed.

clear tt_stack tt table i
