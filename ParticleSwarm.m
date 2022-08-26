rng default

%opt = optimoptions('particleswarm','SwarmSize',100,'HybridFcn',@fmincon);

opt = optimoptions('particleswarm','SwarmSize',100,'Display','iter','PlotFcn', {@pswplotbestf});

lb = [-50,-70];

ub = [50,70];

initial_flag = 0;

for i = 1:15
    %[x,fval,exit_flag,opt] = particleswarm(@(x) benchmark_func(x,10),2,lb,ub,opt);
    %[x,fval,exit_flag,opt] = particleswarm(@(x) benchmark_func(x,10),10,lb,ub,opt);
    %[x,fval,exit_flag,opt] = particleswarm(@(x) benchmark_func(x,11),2,lb,ub,opt);
    [x,fval,exit_flag,opt] = particleswarm(@(x) benchmark_func(x,11),10,lb,ub,opt);
    val(i) = fval;
    %option(i) = opt;
end