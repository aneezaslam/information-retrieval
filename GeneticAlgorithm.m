clc, clear, close all

opt = optimoptions('ga','PlotFcn', {@gaplotbestf, @gaplotdistance});

for i = 1:15

    initial_flag = 0;

    [x,fval,exit_flag,opt] = ga(@(x) benchmark_func(x,10),2,opt);

    %[x,fval,exit_flag,opt] = ga(@(x) benchmark_func(x,10),10,opt);

    %[x,fval,exit_flag,opt] = ga(@(x) benchmark_func(x,10),2,opt);

    %[x,fval,exit_flag,opt] = ga(@(x) benchmark_func(x,10),10,opt);

    val(i) = fval;

    option(i) = opt;

end
