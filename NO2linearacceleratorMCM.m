clear;clc;close all;
rng('shuffle');
tic;

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

try
    parallel.ProfileFactory.createProfile('local');
    fprintf('✅ 并行配置文件重置成功\n');
catch
    fprintf('⚠️ 并行配置文件无需重置，继续执行\n');
end

delete(gcp('nocreate')); 
pause(2);

M_single_std = urel.M1 * M_single_mean;
batch_results = cell(n_batch, 1); 
batch_sizes = zeros(n_batch, 1);  

pool_cores = max(1, floor(cpu_cores / 2)); 
fprintf('CPU物理核心数：%d | 并行核心数：%d | 批次大小：%d | 总批次：%d\n', cpu_cores, pool_cores, batch_size, n_batch);

try
    pool = parpool('local', pool_cores, 'IdleTimeout', 300);
    is_parallel = true;
    fprintf('✅ 并行池开启成功（%d核心），使用并行计算加速\n', pool.NumWorkers);
    
    parfor batch_idx = 1:n_batch
        idx_start = (batch_idx-1)*batch_size + 1;
        idx_end   = min(batch_idx*batch_size, N_sim);
        current_batch_size = idx_end - idx_start + 1;

        M_meas_batch = normrnd(M_single_mean, M_single_std, current_batch_size, n_meas);
        batch_data = mean(M_meas_batch, 2); 

        X_Nk = Nk_mean .* (1 + normrnd(0, urel.M2, current_batch_size, 1));
        X_M3 = batch_data .* (1 + unifrnd(dev_ranges.M3(1), dev_ranges.M3(2), current_batch_size, 1));
        X_M4 = X_M3 .* (1 + unifrnd(dev_ranges.M4_P(1), dev_ranges.M4_P(2), current_batch_size, 1));
        X_Nonlinear = X_M4 .* (1 + unifrnd(dev_ranges.Nonlinear(1), dev_ranges.Nonlinear(2), current_batch_size, 1));
        X_Leakage = X_Nonlinear .* (1 + unifrnd(dev_ranges.Leakage(1), dev_ranges.Leakage(2), current_batch_size, 1));
        X_Src_Stable = X_Leakage .* (1 + unifrnd(dev_ranges.Src_Stable(1), dev_ranges.Src_Stable(2), current_batch_size, 1));
        X_Sw = Sw_air_nominal .* (1 + unifrnd(dev_ranges.Sw(1), dev_ranges.Sw(2), current_batch_size, 1));
        X_Pu = Pu_nominal .* (1 + unifrnd(dev_ranges.Pu(1), dev_ranges.Pu(2), current_batch_size, 1));
        X_Pcel = Pcel_nominal .* (1 + unifrnd(dev_ranges.Pcel(1), dev_ranges.Pcel(2), current_batch_size, 1));
        X_Katt = Katt_nominal .* (1 + unifrnd(dev_ranges.Katt(1), dev_ranges.Katt(2), current_batch_size, 1));
        X_Km = Km_nominal .* (1 + unifrnd(dev_ranges.Km(1), dev_ranges.Km(2), current_batch_size, 1));
        
        t = unifrnd(t_mean - t_half_width, t_mean + t_half_width, current_batch_size, 1);
        P = unifrnd(P_mean - P_half_width, P_mean + P_half_width, current_batch_size, 1);
        T = t + 273.15;
        k_rho_air = (P0 ./ P) .* (T ./ T0);
        
        Dw_batch = X_Src_Stable .* k_rho_air .* X_Nk .* one_minus_g ...
                   .* X_Katt .* X_Km .* X_Sw .* X_Pu .* X_Pcel;

        batch_results{batch_idx} = Dw_batch;
        batch_sizes(batch_idx) = current_batch_size;

        if mod(batch_idx,5)==0
            fprintf('已完成第%d/%d批次\n',batch_idx,n_batch);
        end
    end
catch ME
    is_parallel = false;
    fprintf('⚠️ 并行池开启失败，使用串行计算（错误信息：%s）\n', ME.message);
    
    for batch_idx = 1:n_batch
        idx_start = (batch_idx-1)*batch_size + 1;
        idx_end   = min(batch_idx*batch_size, N_sim);
        current_batch_size = idx_end - idx_start + 1;

        M_meas_batch = normrnd(M_single_mean, M_single_std, current_batch_size, n_meas);
        batch_data = mean(M_meas_batch, 2); 

        X_Nk = Nk_mean .* (1 + normrnd(0, urel.M2, current_batch_size, 1));
        X_M3 = batch_data .* (1 + unifrnd(dev_ranges.M3(1), dev_ranges.M3(2), current_batch_size, 1));
        X_M4 = X_M3 .* (1 + unifrnd(dev_ranges.M4_P(1), dev_ranges.M4_P(2), current_batch_size, 1));
        X_Nonlinear = X_M4 .* (1 + unifrnd(dev_ranges.Nonlinear(1), dev_ranges.Nonlinear(2), current_batch_size, 1));
        X_Leakage = X_Nonlinear .* (1 + unifrnd(dev_ranges.Leakage(1), dev_ranges.Leakage(2), current_batch_size, 1));
        X_Src_Stable = X_Leakage .* (1 + unifrnd(dev_ranges.Src_Stable(1), dev_ranges.Src_Stable(2), current_batch_size, 1));
        X_Sw = Sw_air_nominal .* (1 + unifrnd(dev_ranges.Sw(1), dev_ranges.Sw(2), current_batch_size, 1));
        X_Pu = Pu_nominal .* (1 + unifrnd(dev_ranges.Pu(1), dev_ranges.Pu(2), current_batch_size, 1));
        X_Pcel = Pcel_nominal .* (1 + unifrnd(dev_ranges.Pcel(1), dev_ranges.Pcel(2), current_batch_size, 1));
        X_Katt = Katt_nominal .* (1 + unifrnd(dev_ranges.Katt(1), dev_ranges.Katt(2), current_batch_size, 1));
        X_Km = Km_nominal .* (1 + unifrnd(dev_ranges.Km(1), dev_ranges.Km(2), current_batch_size, 1));
        
        t = unifrnd(t_mean - t_half_width, t_mean + t_half_width, current_batch_size, 1);
        P = unifrnd(P_mean - P_half_width, P_mean + P_half_width, current_batch_size, 1);
        T = t + 273.15;
        k_rho_air = (P0 ./ P) .* (T ./ T0);
        
        Dw_batch = X_Src_Stable .* k_rho_air .* X_Nk .* one_minus_g ...
                   .* X_Katt .* X_Km .* X_Sw .* X_Pu .* X_Pcel;

        batch_results{batch_idx} = Dw_batch;
        batch_sizes(batch_idx) = current_batch_size;

        if mod(batch_idx,5)==0
            fprintf('已完成第%d/%d批次\n',batch_idx,n_batch);
        end
    end
end

Dw_sample = zeros(N_sim, 1);
current_pos = 1;
for batch_idx = 1:n_batch
    batch_data = batch_results{batch_idx};
    batch_size = batch_sizes(batch_idx);
    Dw_sample(current_pos:current_pos+batch_size-1) = batch_data;
    current_pos = current_pos + batch_size;
end

mean_seq    = zeros(1, total_check);
std_seq     = zeros(1, total_check);
lower95_seq = zeros(1, total_check);
upper95_seq = zeros(1, total_check);
crit1_seq   = zeros(1, total_check);
crit2_seq   = zeros(1, total_check);
crit3_seq   = zeros(1, total_check);
crit4_seq   = zeros(1, total_check);

fprintf('\n计算核心统计量，总节点数：%d\n', total_check);
progress_bar = waitbar(0, '统计量计算中...');

for node_idx = 1:total_check
    n = check_points(node_idx);
    Dw_sub = Dw_sample(1:n);

    mean_seq(node_idx)    = mean(Dw_sub);
    std_seq(node_idx)     = std(Dw_sub);
    lower95_seq(node_idx) = quantile(Dw_sub, 0.025);
    upper95_seq(node_idx) = quantile(Dw_sub, 0.975);

    waitbar(node_idx/total_check, progress_bar, sprintf('进度：%.1f%%', node_idx/total_check*100));
end
close(progress_bar);
fprintf('统计量计算完成！\n');

for node_idx = 2:total_check
    crit1_seq(node_idx) = 2*std(mean_seq(1:node_idx));
    crit2_seq(node_idx) = 2*std(std_seq(1:node_idx));
    crit3_seq(node_idx) = 2*std(lower95_seq(1:node_idx));
    crit4_seq(node_idx) = 2*std(upper95_seq(1:node_idx));
end
crit1_seq(1) = NaN; crit2_seq(1) = NaN; crit3_seq(1) = NaN; crit4_seq(1) = NaN;

convergence_steps = ones(1, n_tols)*N_sim;
tols_crit1 = zeros(1, n_tols);
tols_crit2 = zeros(1, n_tols);
tols_crit3 = zeros(1, n_tols);
tols_crit4 = zeros(1, n_tols);
tols_conv_mean = zeros(1, n_tols);
tols_conv_std = zeros(1, n_tols);
tols_conv_l95 = zeros(1, n_tols);
tols_conv_u95 = zeros(1, n_tols);

fprintf('\n===== 容差收敛判定（保留2×标准差+绝对值容差） =====\n');
fprintf('单次校准测量次数：%d\n',n_meas);
for tol_idx = 1:n_tols
    tol_W = tolerances_W(tol_idx);
    converged_idx = total_check;
    is_converged = false;

    for node_idx = 2:total_check
        if ~isnan(crit1_seq(node_idx)) && ~isnan(crit2_seq(node_idx)) && ...
           ~isnan(crit3_seq(node_idx)) && ~isnan(crit4_seq(node_idx)) && ...
           crit1_seq(node_idx) < tol_W && crit2_seq(node_idx) < tol_W && ...
           crit3_seq(node_idx) < tol_W && crit4_seq(node_idx) < tol_W
            converged_idx = node_idx;
            is_converged = true;
            break;
        end
    end

    convergence_steps(tol_idx) = check_points(converged_idx);
    tols_crit1(tol_idx) = crit1_seq(converged_idx);
    tols_crit2(tol_idx) = crit2_seq(converged_idx);
    tols_crit3(tol_idx) = crit3_seq(converged_idx);
    tols_crit4(tol_idx) = crit4_seq(converged_idx);
    tols_conv_mean(tol_idx) = mean_seq(converged_idx);
    tols_conv_std(tol_idx) = std_seq(converged_idx);
    tols_conv_l95(tol_idx) = lower95_seq(converged_idx);
    tols_conv_u95(tol_idx) = upper95_seq(converged_idx);

    if is_converged
        status = '✅ 达标';
    else
        status = '❌ 未达标';
    end
    fprintf('容差%.3f mGy：%s | 实际crit1=%.6f, crit2=%.6f, crit3=%.6f, crit4=%.6f | 达标样本量=%d\n', ...
        tol_W, status, tols_crit1(tol_idx), tols_crit2(tol_idx), tols_crit3(tol_idx), tols_crit4(tol_idx), convergence_steps(tol_idx));
end

Dw_mean_final = mean(Dw_sample);
Dw_std_final  = std(Dw_sample);
Dw_urel_final = Dw_std_final / Dw_mean_final;
Dw_ci95_final = [quantile(Dw_sample,0.025), quantile(Dw_sample,0.975)];
k_rho_air_mean = (P0 / P_mean) * (T_mean / T0);
Dw_theory = M_single_mean * Nk_mean * one_minus_g * Katt_nominal * Km_nominal ...
            * Sw_air_nominal * Pu_nominal * Pcel_nominal * k_rho_air_mean;

fprintf('\n=== 核心结果汇总 ===\n');
fprintf('1. 水中吸收剂量均值：%.4f mGy\n', Dw_mean_final);
fprintf('2. 标准不确定度：%.4f mGy\n', Dw_std_final);
fprintf('3. 相对标准不确定度：%.4f%%\n', Dw_urel_final*100);
fprintf('4. 95%%包含区间：[%.4f, %.4f] mGy\n', Dw_ci95_final(1), Dw_ci95_final(2));
fprintf('5. 理论均值（无波动）：%.4f mGy\n', Dw_theory);

fprintf('\n=== 容差收敛详细结果 ===\n');
fprintf('┌──────────┬────────────┬───────────────┬─────────────┬────────────┬────────────┬──────────┬──────────┬──────────┬──────────┐\n');
fprintf('│ 容差(mGy)│ 达标样本量 │ 达标均值(mGy) │ 达标std(mGy)│ 95%%左(mGy) │ 95%%右(mGy) │  crit1   │  crit2   │  crit3   │  crit4   │\n');
fprintf('├──────────┼────────────┼───────────────┼─────────────┼────────────┼────────────┼──────────┼──────────┼──────────┼──────────┤\n');
for tol_idx = 1:n_tols
    fprintf('│ %.3f     │ %-8d   │ %-11.4f   │ %-10.4f   │ %-9.4f   │ %-9.4f   │ %.6f │ %.6f │ %.6f │ %.6f │\n', ...
        tolerances_W(tol_idx), convergence_steps(tol_idx), ...
        tols_conv_mean(tol_idx), tols_conv_std(tol_idx), ...
        tols_conv_l95(tol_idx), tols_conv_u95(tol_idx), ...
        tols_crit1(tol_idx), tols_crit2(tol_idx), tols_crit3(tol_idx), tols_crit4(tol_idx));
end
fprintf('└──────────┴────────────┴───────────────┴─────────────┴────────────┴────────────┴──────────┴──────────┴──────────┴──────────┘\n');

figure('Color','w','Position',[100,100,800,600]);
hold on;
bin_width = 0.05;
dw_min = floor(min(Dw_sample)/bin_width) * bin_width;
dw_max = ceil(max(Dw_sample)/bin_width) * bin_width;
bin_edges = dw_min:bin_width:dw_max;
[bin_counts, ~] = histcounts(Dw_sample, bin_edges);
bin_centers = (bin_edges(1:end-1) + bin_edges(2:end))/2;
h_hist = bar(bin_centers, bin_counts, 'FaceAlpha',0.8);
h_hist.FaceColor = [0.1 0.5 0.8];
h_hist.EdgeColor = 'white';
h_hist.LineWidth = 0.5;
x_fit = linspace(Dw_mean_final-4*Dw_std_final, Dw_mean_final+4*Dw_std_final, 1000);
y_fit = normpdf(x_fit, Dw_mean_final, Dw_std_final) * N_sim * bin_width;
plot(x_fit, y_fit, 'r-', 'LineWidth',1.5);
xline(Dw_ci95_final(1), 'g--', 'LineWidth',1.5);
xline(Dw_ci95_final(2), 'k--', 'LineWidth',1.5);
xline(Dw_theory, 'm-.', 'LineWidth',1.5);
title('光子束水中吸收剂量分布直方图（固定0.05间隔 | 频次）');
xlabel('Dw (mGy)'); ylabel('频次');
legend('蒙特卡洛样本频次分布','正态拟合线','2.5%分位数','97.5%分位数','理论均值','Location','best'); 
grid on; hold off;

hist_table = table(bin_centers', bin_counts', 'VariableNames', {'Dw区间中点(mGy)','频次'});
fit_table = table(x_fit', y_fit', 'VariableNames', {'Dw横坐标(mGy)','拟合频次'});
writetable(hist_table, '直方图数据_固定0.05间隔_频次.csv', 'WriteVariableNames', true, 'Encoding', 'UTF-8');
writetable(fit_table, '正态拟合数据_固定0.05间隔_频次.csv', 'WriteVariableNames', true, 'Encoding', 'UTF-8');
fprintf('✅ 图1数据（频次版）已保存\n');

figure('Color','w','Position',[100,100,800,600]);
hold on;
plot(check_points(2:end), crit1_seq(2:end), 'r-', 'LineWidth',1.5);
plot(check_points(2:end), crit2_seq(2:end), 'b-', 'LineWidth',1.5);
plot(check_points(2:end), crit3_seq(2:end), 'g-', 'LineWidth',1.5);
plot(check_points(2:end), crit4_seq(2:end), 'k-', 'LineWidth',1.5);
for tol_idx = 1:n_tols
    yline(tolerances_W(tol_idx), ':','Color',[0.5,0.5,0.5]);
end
set(gca,'XScale','log'); set(gca,'YScale','log');
ylim([1e-4, 1]);
title('四大容差指标随模拟步数变化');
xlabel('模拟步数（对数）'); ylabel('容差值（mGy）');
legend('crit1（均值2倍标准差）','crit2（标准差2倍标准差）','crit3（左分位数2倍标准差）','crit4（右分位数2倍标准差）','Location','best'); 
grid on; hold off;

figure('Color','w','Position',[100,100,800,600]);
plot(check_points, mean_seq, 'm-', 'LineWidth',1.5);
hold on;
yline(Dw_theory, 'r--', 'LineWidth',1.5);
set(gca,'XScale','log'); 
title('剂量均值收敛曲线');
xlabel('模拟次数（对数）'); ylabel('Dw (mGy)'); 
legend('模拟均值','理论均值'); grid on; hold off;

figure('Color','w','Position',[100,100,800,600]);
plot(check_points, std_seq, 'c-', 'LineWidth',1.5);
hold on; 
set(gca,'XScale','log'); 
title('标准不确定度收敛曲线');
xlabel('模拟次数（对数）'); ylabel('标准不确定度（mGy）'); 
legend; grid on; hold off;

figure('Color','w','Position',[100,100,800,600]);
qqplot(Dw_sample); 
standardized_samples = (Dw_sample - Dw_mean_final) / Dw_std_final;
[h_test, p_val, ks_stat] = kstest(standardized_samples, 'CDF', [sort(standardized_samples), normcdf(sort(standardized_samples))], 'Alpha', 0.05);
test_text = sprintf('KS检验：统计量=%.4f, p值=%.4f\nH0: 符合正态（p>0.05接受）', ks_stat, p_val);
text(0.05, 0.95, test_text, 'Units','normalized', 'BackgroundColor','white', 'EdgeColor','black');
title('Q-Q图（正态性评估）'); xlabel('理论分位数'); ylabel('样本分位数'); grid on;

figure('Color','w','Position',[100,100,800,500]);
set(gca, 'Visible', 'off');
text_content = sprintf(...
'=====================\n  水中吸收剂量核心指标汇总\n  （TRS-277修正，5次测量）\n=====================\n1. 均值：%.4f mGy\n2. 标准不确定度：%.4f mGy\n3. 相对标准不确定度：%.6f%%\n4. 95%%包含区间：[%.4f , %.4f] mGy\n=====================',...
Dw_mean_final, Dw_std_final, Dw_urel_final*100, Dw_ci95_final(1), Dw_ci95_final(2));
text(0.5, 0.5, text_content, 'Units','normalized', 'HorizontalAlignment','center', 'VerticalAlignment','middle', 'FontSize',14, 'FontWeight','bold');

if is_parallel
    delete(pool);
end

toc;