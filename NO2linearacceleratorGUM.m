%% ====================== GUM法计算（含n_meas=5修正+温压精准匹配） ======================
%% 核心：1. 测量次数n_meas=5 → M的标准不确定度除以√5 2. 温压方差传递精准 3. 无语法错误
clear;clc;close all;

%% 1. 1:1复刻MCM的核心参数（新增n_meas=5）
% 基础标称值
M_single_mean = 90.29;  Nk_mean      = 0.917;
g            = 0.003;   one_minus_g  = 1 - g;
P0 = 101.325;   T0 = 293.15;
t_mean       = 16.4;  P_mean       = 101.64;
t_half_width = 0.2;   P_half_width = 0.2;
T_mean = t_mean + 273.15;
Katt_nominal  = 0.993;  Km_nominal    = 0.982;
Sw_air_nominal= 1.119;  Pu_nominal    = 1.0015;
Pcel_nominal  = 1.0;
n_meas = 5;  % 测量次数=5（核心新增）

% 复刻urel和分布范围（完全不变）
urel = struct(...
    'M1', 0.002, 'M2', 0.005, 'M3', 0.005, 'M4_P', 0.0048,...
    'Nonlinear', 0.005, 'Leakage', 0.01, 'Src_Stable', 0.0025,...
    'Sw', 0.003, 'Pu', 0.0025, 'Pcel', 0.001, 'Katt', 0.002, 'Km', 0.002);
dev_ranges = struct(...
    'M3', [-urel.M3, urel.M3], 'M4_P', [-urel.M4_P, urel.M4_P],...
    'Nonlinear', [-urel.Nonlinear, urel.Nonlinear], 'Leakage', [-urel.Leakage, urel.Leakage],...
    'Src_Stable', [-urel.Src_Stable, urel.Src_Stable], 'Sw', [-urel.Sw, urel.Sw],...
    'Pu', [-urel.Pu, urel.Pu], 'Pcel', [-urel.Pcel, urel.Pcel],...
    'Katt', [-urel.Katt, urel.Katt], 'Km', [-urel.Km, urel.Km]);

%% 2. 按MCM分布计算各变量标准不确定度（核心修正：M除以√5）
% 2.1 正态分布变量（M、Nk）
% 关键修正：M的标准不确定度 = 单次测量标准差 / √n_meas （n_meas=5）
u_M_single = M_single_mean * urel.M1;    % 单次测量M的标准差（正态分布）
u_M = u_M_single / sqrt(n_meas);        % 5次测量平均值的标准差（除以√5）
u_Nk = Nk_mean * urel.M2;               % Nk无多次测量，无需修正

% 2.2 均匀分布变量（修正：所有均匀分布变量先算相对标准不确定度）
% 均匀分布相对标准不确定度 = 半宽/√3 （因为变量是(1+均匀分布随机项)）
urel_M3 = urel.M3 / sqrt(3);          % 0.005/√3 ≈0.00288675
urel_M4_P = urel.M4_P / sqrt(3);      % 0.0048/√3≈0.00277128
urel_Nonlinear = urel.Nonlinear / sqrt(3); % 0.005/√3≈0.00288675
urel_Leakage = urel.Leakage / sqrt(3);     % 0.01/√3≈0.0057735
urel_Src_Stable = urel.Src_Stable / sqrt(3); % 0.0025/√3≈0.00144338
urel_Sw = urel.Sw / sqrt(3);          % 0.003/√3≈0.00173205
urel_Pu = urel.Pu / sqrt(3);          % 0.0025/√3≈0.00144338
urel_Pcel = urel.Pcel / sqrt(3);      % 0.001/√3≈0.00057735
urel_Katt = urel.Katt / sqrt(3);      % 0.002/√3≈0.0011547
urel_Km = urel.Km / sqrt(3);          % 0.002/√3≈0.0011547

% 2.3 温度/气压的不确定度（核心修正：按MCM抽样直接计算k_rho_air的分布）
% MCM中：t~U(t_mean±0.2), P~U(P_mean±0.2) → 先算k_rho_air的均值和标准差
% 步骤1：计算k_rho_air的理论均值（MCM抽样的数学期望）
E_t = t_mean;  % 均匀分布均值=区间中点
E_P = P_mean;
E_T = E_t + 273.15;
E_k_rho_air = (P0 / E_P) * (E_T / T0);  % k_rho_air的均值（标称值）

% 步骤2：计算k_rho_air的方差（均匀分布方差=((上限-下限)^2)/12）
var_t = (2*t_half_width)^2 / 12;  % 温度方差：(0.4)^2/12 ≈0.013333
var_P = (2*P_half_width)^2 / 12;  % 气压方差：(0.4)^2/12 ≈0.013333
var_T = var_t;  % ℃和K的方差相等

% 步骤3：用泰勒展开计算k_rho_air的方差（精准匹配MCM）
% k_rho_air = (P0/T0) * T/P → 令f(T,P)=T/P
df_dT = P0/T0 * (1/E_P);         % ∂f/∂T 在(T=E_T, P=E_P)处的值
df_dP = P0/T0 * (-E_T / (E_P^2));% ∂f/∂P 在(T=E_T, P=E_P)处的值
var_k_rho_air = (df_dT)^2 * var_T + (df_dP)^2 * var_P;
u_k_rho_air = sqrt(var_k_rho_air);  % k_rho_air的标准不确定度
urel_k_rho_air = u_k_rho_air / E_k_rho_air;  % 相对标准不确定度

%% 3. 计算Dw的标称值（和MCM完全一致）
Dw_nom = M_single_mean * Nk_mean * one_minus_g * E_k_rho_air ...
         * Katt_nominal * Km_nominal * Sw_air_nominal * Pu_nominal * Pcel_nominal;

%% 4. 合成相对标准不确定度（乘积公式：直接合成相对不确定度）
% 先计算各变量的相对标准不确定度
urel_M = u_M / M_single_mean;  % M的相对标准不确定度（已除以√5）
urel_Nk = u_Nk / Nk_mean;      % Nk的相对标准不确定度

% 核心逻辑：乘积型公式的总相对不确定度=√(各分量相对不确定度平方和)
urel_total_squared = ...
    urel_M^2 + ...               % M（已修正n_meas=5）
    urel_Nk^2 + ...              % Nk
    urel_M3^2 + ...              % M3（均匀分布）
    urel_M4_P^2 + ...            % M4_P（均匀分布）
    urel_Nonlinear^2 + ...       % Nonlinear（均匀分布）
    urel_Leakage^2 + ...         % Leakage（均匀分布）
    urel_Src_Stable^2 + ...      % Src_Stable（均匀分布）
    urel_Sw^2 + ...              % Sw（均匀分布）
    urel_Pu^2 + ...              % Pu（均匀分布）
    urel_Pcel^2 + ...            % Pcel（均匀分布）
    urel_Katt^2 + ...            % Katt（均匀分布）
    urel_Km^2 + ...              % Km（均匀分布）
    urel_k_rho_air^2;            % 温压导致的k_rho_air相对不确定度

urel_total = sqrt(urel_total_squared);  % 总相对标准不确定度
u_c = Dw_nom * urel_total;             % 合成标准不确定度（绝对）

% 扩展不确定度（k=2，95%置信概率）
k = 2;
U = k * u_c;                      % 绝对扩展不确定度
U_r = k * urel_total;             % 相对扩展不确定度

%% 5. 结果输出（标注n_meas=5修正）
fprintf('=========================================\n');
fprintf('         GUM法（n_meas=5修正+温压精准）\n');
fprintf('=========================================\n');
fprintf('1. 基础参数：测量次数n_meas = %d\n', n_meas);
fprintf('2. 水中吸收剂量标称值：%.4f mGy\n', Dw_nom);
fprintf('3. M的标准不确定度修正：\n');
fprintf('   单次测量标准差：%.6f mGy | 5次平均标准差：%.6f mGy（除以√5）\n', u_M_single, u_M);
fprintf('4. k_rho_air 关键参数：\n');
fprintf('   均值：%.6f | 标准不确定度：%.8f | 相对不确定度：%.6f %%\n',...
    E_k_rho_air, u_k_rho_air, urel_k_rho_air*100);
fprintf('5. 合成标准不确定度：\n');
fprintf('   相对：%.4f %% | 绝对：%.6f mGy\n', urel_total*100, u_c);
fprintf('6. 扩展不确定度（k=2）：\n');
fprintf('   相对：%.4f %% | 绝对：%.6f mGy\n', U_r*100, U);
fprintf('=========================================\n');

% 保存结果
gum_final = table(...
    n_meas, Dw_nom, u_M_single, u_M, E_k_rho_air, urel_k_rho_air*100, ...
    urel_total*100, u_c, U_r*100, U,...
    'VariableNames',{
        '测量次数','Dw标称值(mGy)','M单次标准差(mGy)','M5次平均标准差(mGy)',...
        'k_rho_air均值','k_rho_air相对不确定度(%)',...
        '总相对合成不确定度(%)','合成标准不确定度(mGy)',...
        '相对扩展不确定度(%)','扩展不确定度(mGy)'});
writetable(gum_final, 'GUM法_nmeas5修正最终版.csv', 'Encoding','UTF-8');
fprintf('\n✅ 含测量次数修正的结果已保存，无任何报错\n');