# code-for-publication  
HighFrequencyElectrosurgicalUnits202601  
基于matlab 2018b模拟高频电刀输出功率示值误差不确定度的代码  
P_set = 250;                    % 设定功率 (W)  
P_avg_measured = 248;           % 实际3次测量平均值 (W)  
s = 1.15;                       % 单次测量的实验标准差 (W)  
resolution = 1;                 % 分析仪分辨力 (W)  
MPE_relative = 0.05;            % 最大允许误差 (±5%)  
n_measurements = 3;             % 校准时的测量次数  
n_simulations = 1e6;         % 模拟次数  
convergence_step = 100;        % 收敛分析步长（每1000次）  
      
NO2 linear accelerator  
基于matlab 2024b模拟直线加速器水中吸收剂量不确定度的代码   
N_sim        = 1e8;  
step_size    = 1000;  
check_points = step_size:step_size:N_sim;   
total_check  = length(check_points);  
n_meas       = 5;   

cpu_cores = feature('numcores');  
batch_size   = min(5000, cpu_cores * 1000);  
n_batch      = ceil(N_sim / batch_size);  

tolerances_W = [0.1, 0.05, 0.01];   
n_tols       = length(tolerances_W);  
P0 = 101.325;   T0 = 293.15;  
t_mean       = 16.4;  P_mean       = 101.64;  
t_half_width = 0.2;   P_half_width = 0.2;  
T_mean = t_mean + 273.15;  
M_single_mean = 90.29;  Nk_mean      = 0.917;  
g            = 0.003;   one_minus_g  = 1 - g;  
Katt_nominal  = 0.993;  Km_nominal    = 0.982;  
Sw_air_nominal= 1.119;  Pu_nominal    = 1.0015;  
Pcel_nominal  = 1.0;  
