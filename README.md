# code-for-publication
HighFrequencyElectrosurgicalUnits202601
基于matlab 2018b模拟高频电刀输出功率示值误差的代码
P_set = 250;                    % 设定功率 (W)
P_avg_measured = 248;           % 实际3次测量平均值 (W)
s = 1.15;                       % 单次测量的实验标准差 (W)
resolution = 1;                 % 分析仪分辨力 (W)
MPE_relative = 0.05;            % 最大允许误差 (±5%)
n_measurements = 3;             % 校准时的测量次数
n_simulations = 1e6;         % 模拟次数
convergence_step = 100;        % 收敛分析步长（每1000次）
