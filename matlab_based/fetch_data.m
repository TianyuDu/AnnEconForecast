% This module is to fetch data from Fred Database.


if modelPara.dataName == "UNRATE"
	fprintf("Fetching UNRATE data...\n");
	fileName = "UNRATE.csv";
	data.main = csvread(fileName, 1, 1)';
	
	data.d1 = diff(data.main);  % First order difference in time series.
	data.d2 = diff(data.main, 2);  % Second order difference in time series.
	data.len= length(data.d2);  % Data length.
	
	data.main = data.main(3: end);
	data.d1 = data.d1(2: end);
	data.stack = [data.main; data.d1; data.d2];
end

if modelPara.dataName == "GDP"
	fprintf("Fetching GDP data...\n");
	fileName = "GDP.csv";
	data.main = csvread(fileName, 1, 1)';
end

if modelPara.dataName == "DEXUSEU"
	load("SampleData.mat")
	data.main = DEXUSEU';
	data.d1 = diff(data.main);
	data.d2 = diff(data.main, 2);
	data.len = length(data.d2);
	
	data.main = data.main(3: end);
	data.d1 = data.d1(2: end);
	data.stack = [data.main; data.d1; data.d2];
end