clear all; close all; clc;
%% ====================== 1. 输入参数（优化：降低模拟次数，避免内存溢出） ======================
P_set = 250;                    % 设定功率 (W)
P_avg_measured = 248;           % 实际3次测量平均值 (W)
s = 1.15;                       % 单次测量的实验标准差 (W)
resolution = 1;                 % 分析仪分辨力 (W)
MPE_relative = 0.05;            % 最大允许误差 (±5%)
n_measurements = 3;             % 校准时的测量次数
n_simulations = 1e6;         % 模拟次数
convergence_step = 100;        % 收敛分析步长（每1000次）

%% ====================== 2. 蒙特卡洛核心模拟  ======================

E_actual = P_set - P_avg_measured;
P_avg_simulated = zeros(n_simulations, 1);
individual_measurements = zeros(n_simulations, n_measurements);

for i = 1:n_simulations
    P_three_measurements = zeros(1, n_measurements);
    MPE_error = 0;
    for j = 1:n_measurements
        repeatability_error = s * randn();
        resolution_error = resolution * (rand() - 0.5);
        if j == 1
            MPE_absolute = P_avg_measured * MPE_relative;
            MPE_error = 2 * MPE_absolute * (rand() - 0.5);
        end
        P_single = P_avg_measured + repeatability_error + resolution_error + MPE_error;
        P_three_measurements(j) = P_single;
    end
    P_avg_simulated(i) = mean(P_three_measurements);
    individual_measurements(i, :) = P_three_measurements;
end
E_simulated = P_set - P_avg_simulated;

convergence_points = floor(n_simulations / convergence_step);
convergence_points = max(convergence_points, 1);
uncertainty_convergence = zeros(convergence_points, 1);
mean_convergence = zeros(convergence_points, 1);
simulation_counts = zeros(convergence_points, 1);
relative_error = zeros(convergence_points, 1);

for i = 1:convergence_points
    n_current = min(i * convergence_step, n_simulations);
    simulation_counts(i) = n_current;
    current_E = E_simulated(1:n_current);
    uncertainty_convergence(i) = std(current_E);
    mean_convergence(i) = mean(current_E);
    if mod(i, 20) == 0
        fprintf('模拟%d次，标准不确定度：%.4f W\n',n_current,uncertainty_convergence(i));
    end
end

final_uncertainty = std(E_simulated);
final_mean = mean(E_simulated);
E_95_lower = prctile(E_simulated, 2.5);
E_95_upper = prctile(E_simulated, 97.5);

final_uncertainty = max(final_uncertainty, 1e-6);
relative_error = abs(uncertainty_convergence - final_uncertainty) / final_uncertainty * 100;

thresholds = [5, 2, 1, 0.5, 0.2, 0.1];
convergence_points_needed = ones(size(thresholds)) * n_simulations;
for i = 1:length(thresholds)
    below_threshold = find(relative_error <= thresholds(i), 1, 'first');
    if ~isempty(below_threshold)
        convergence_points_needed(i) = simulation_counts(below_threshold);
    end
end

u1 = s / sqrt(n_measurements);
u2 = (resolution / 2) / sqrt(3);
u3 = (P_avg_measured * MPE_relative) / sqrt(3);
uc = sqrt(u1^2 + u2^2 + u3^2);

contrib1 = (u1^2 / uc^2) * 100;
contrib2 = (u2^2 / uc^2) * 100;
contrib3 = (u3^2 / uc^2) * 100;

u_components = [u1, u2, u3];
u_names = {'重复性','分辨力','MPE'}; 
u_contrib = [contrib1, contrib2, contrib3];

fprintf('\n--- 不确定度分量贡献分析 ---\n');
fprintf('重复性u1：%.4f W（贡献%.2f%%）\n',u1,contrib1);
fprintf('分辨力u2：%.4f W（贡献%.2f%%）\n',u2,contrib2);
fprintf('MPE u3：%.4f W（贡献%.2f%%）\n',u3,contrib3);
fprintf('合成标准不确定度uc（方和根）：%.4f W\n',uc);
fprintf('蒙特卡洛模拟合成不确定度：%.4f W（对比验证）\n',final_uncertainty);

fprintf('\n=== 示值误差分位值结果 ===\n');
fprintf('2.5%%分位值（E_95_lower）：%.3f W\n', E_95_lower);
fprintf('97.5%%分位值（E_95_upper）：%.3f W\n', E_95_upper);
fprintf('\n=== 不确定度评定结果 ===\n');
fprintf('示值误差均值：%.3f W，合成标准不确定度：%.3f W\n',final_mean,final_uncertainty);
fprintf('95%包含区间：[%.3f, %.3f] W，扩展不确定度(k=2)：%.3f W\n',E_95_lower,E_95_upper,2*final_uncertainty);

set(0,'DefaultFigurePosition',[100,100,800,600]);
set(0,'DefaultFigurePaperPositionMode','auto');
set(0,'DefaultAxesFontSize',12);
set(0,'DefaultTextFontSize',10);

figure(1); clf;
h_hist = histogram(E_simulated, 80, 'Normalization','pdf','FaceColor',[0.2,0.6,0.8],'FaceAlpha',0.7);
hold on; grid on;
y_limits = ylim; 

h_mean = plot([final_mean, final_mean], y_limits, 'r-', 'LineWidth',2);
plot([E_95_lower, E_95_lower], y_limits, 'm--', 'LineWidth',1.5);
h_95 = plot([E_95_upper, E_95_upper], y_limits, 'm--', 'LineWidth',1.5);

text(E_95_lower, y_limits(2)*0.9, '2.5%分位数','HorizontalAlignment','right');
text(E_95_upper, y_limits(2)*0.9, '97.5%分位数','HorizontalAlignment','left');
text(final_mean, y_limits(2)*0.8, sprintf('均值=%.3fW',final_mean),'HorizontalAlignment','center');

x_fit = linspace(min(E_simulated), max(E_simulated), 1000); 
y_fit = normpdf(x_fit, final_mean, final_uncertainty);      
h_norm = plot(x_fit, y_fit, 'k-', 'LineWidth',2.5);                 

xlabel('示值误差 (W)'); ylabel('概率密度');
title('示值误差概率分布 - 蒙特卡洛法（含标准正态分布拟合）');

legend([h_hist, h_mean, h_95, h_norm], ...
       '蒙特卡洛概率分布','均值','95%包含区间','标准正态分布拟合线',...
       'Location','best');
saveas(gcf, '1_示值误差概率分布_含正态拟合线.png','png');

figure(21); clf; set(gcf,'Position',[200,600,600,400]);
plot(simulation_counts,uncertainty_convergence,'b-', 'LineWidth',2);
hold on; grid on; x_limits = xlim;
plot(x_limits, [final_uncertainty, final_uncertainty], 'r--','LineWidth',2);
xlabel('模拟次数'); ylabel('标准不确定度 (W)'); 
title('标准不确定度收敛过程');
legend('收敛过程',sprintf('最终值:%.4fW',final_uncertainty),'Location','southeast');
saveas(gcf, '2a_标准不确定度收敛.png','png');

figure(22); clf; set(gcf,'Position',[300,500,600,400]);
semilogx(simulation_counts,relative_error,'r-','LineWidth',2);
hold on; grid on; x_limits2 = xlim;
plot(x_limits2, [1, 1], 'g--','LineWidth',1.5);
plot(x_limits2, [0.5, 0.5], 'g--','LineWidth',1.5);
xlabel('模拟次数'); ylabel('相对误差 (%)'); 
title('相对收敛误差分析');
legend('相对误差','1%/0.5%阈值','Location','northeast');
saveas(gcf, '2b_相对收敛误差.png','png');

figure(23); clf; set(gcf,'Position',[400,400,600,400]);
plot(simulation_counts,mean_convergence,'g-','LineWidth',2);
hold on; grid on; x_limits3 = xlim;
plot(x_limits3, [final_mean, final_mean], 'r--','LineWidth',2);
xlabel('模拟次数'); ylabel('示值误差均值 (W)'); 
title('均值收敛过程');
legend('收敛过程',sprintf('最终值:%.4fW',final_mean),'Location','southeast');
saveas(gcf, '2c_均值收敛.png','png');

figure(24); clf; set(gcf,'Position',[500,300,600,400]);
loglog(simulation_counts,relative_error,'b-','LineWidth',2);
hold on; grid on;
theory_convergence = 100 ./ sqrt(simulation_counts/simulation_counts(1));
loglog(simulation_counts,theory_convergence,'r--','LineWidth',1.5);
xlabel('模拟次数'); ylabel('相对误差 (%)'); 
title('收敛速度分析（1/√N理论参考）');
legend('实际收敛','理论~1/√N','Location','northeast');
saveas(gcf, '2d_收敛速度分析.png','png');

figure(31); clf; set(gcf,'Position',[100,550,600,400]);
bar(thresholds,convergence_points_needed,'FaceColor',[0.7,0.2,0.2]);
grid on; xlabel('收敛阈值 (%)'); ylabel('所需模拟次数'); 
title('收敛阈值分析');
for i=1:length(thresholds)
    text(i,convergence_points_needed(i)*1.05,sprintf('%d',convergence_points_needed(i)),'Horiz','center');
end
saveas(gcf, '3a_收敛阈值分析.png','png');

figure(32); clf; set(gcf,'Position',[200,450,600,400]);
early_p = min(20,length(simulation_counts));
plot(simulation_counts(1:early_p),uncertainty_convergence(1:early_p),'bo-','LineWidth',2);
hold on; grid on; x_limits4 = xlim;
plot(x_limits4, [final_uncertainty, final_uncertainty], 'r--','LineWidth',2);
xlabel('模拟次数'); ylabel('标准不确定度 (W)'); 
title('早期收敛分析（前2万次）');
saveas(gcf, '3b_早期收敛分析.png','png');

figure(33); clf; set(gcf,'Position',[300,350,600,400]);
if length(simulation_counts)>10
    last_p = 10; 
    start_idx = length(simulation_counts)-last_p+1;
    plot(simulation_counts(start_idx:end),uncertainty_convergence(start_idx:end),'mo-','LineWidth',2);
    hold on; x_limits5 = xlim;
    plot(x_limits5, [final_uncertainty, final_uncertainty], 'r--','LineWidth',2);
    last_fluc = range(uncertainty_convergence(start_idx:end))/2;
    text(0.5,0.9,sprintf('波动±%.4fW',last_fluc),'Units','normalized');
else
    text(0.5,0.5,'模拟次数不足','Units','normalized','Horiz','center');
end
grid on; xlabel('模拟次数'); ylabel('标准不确定度 (W)'); 
title('后期稳定性分析（最后1万次）');
saveas(gcf, '3c_后期稳定性分析.png','png');

figure(34); clf; set(gcf,'Position',[400,250,600,400]);
semilogy(simulation_counts,relative_error,'k-','LineWidth',2);
hold on; grid on; x_limits6 = xlim;
plot(x_limits6, [1, 1], 'g--','LineWidth',1.5);
plot(x_limits6, [0.5, 0.5], 'g--','LineWidth',1.5);
xlabel('模拟次数'); ylabel('相对误差 (%)'); 
title('相对误差收敛分析');
saveas(gcf, '3d_相对误差收敛.png','png');

figure(35); clf; set(gcf,'Position',[500,150,600,400]);
idx_1p = find(relative_error<=1,1,'first');
if ~isempty(idx_1p)
    plot(simulation_counts(1:idx_1p),relative_error(1:idx_1p),'ro-','LineWidth',2);
    hold on; y_limits5 = ylim;
    plot([simulation_counts(idx_1p), simulation_counts(idx_1p)], y_limits5, 'b--','LineWidth',2);
    text(0.5,0.9,sprintf('达到1%%：%d次',simulation_counts(idx_1p)),'Units','normalized');
else
    text(0.5,0.5,'未达到1%阈值','Units','normalized','Horiz','center');
end
grid on; xlabel('模拟次数'); ylabel('相对误差 (%)'); 
title('收敛到1%阈值过程分析');
saveas(gcf, '3e_收敛到1%阈值过程.png','png');

figure(36); clf; set(gcf,'Position',[600,50,600,400]);
axis off;
text(0.1,0.8,sprintf('最终标准不确定度：%.4f W',final_uncertainty),'FontSize',12);
text(0.1,0.6,sprintf('总模拟次数：%d',n_simulations),'FontSize',12);
text(0.1,0.4,sprintf('最终相对误差：%.3f%%',relative_error(end)),'FontSize',12);
text(0.1,0.2,'收敛分析总结','FontSize',14,'FontWeight','bold');
title('收敛分析总结');
saveas(gcf, '3f_收敛分析总结.png','png');

fprintf('图2和图3的子图已分别单独绘制并保存！\n');

figure(4); clf;
colors_line = lines(3);
for j=1:n_measurements
    histogram(individual_measurements(:,j),50,'Normalization','pdf','FaceColor',colors_line(j,:),'FaceAlpha',0.6,'DisplayName',sprintf('第%d次测量',j));
    hold on;
end
histogram(P_avg_simulated,50,'Normalization','pdf','FaceColor','k','FaceAlpha',0.7,'DisplayName','3次平均值');
grid on; xlabel('功率测量值 (W)'); ylabel('概率密度'); title('单次测量与平均值分布对比');
legend('Location','best');
saveas(gcf, '4_单次与平均值分布.png','png');

figure(5); clf;
[F, x] = ecdf(E_simulated);
plot(x, F, 'b-', 'LineWidth',2); hold on; grid on;
plot([E_95_lower,E_95_upper],[0.025,0.975],'ro','MarkerSize',8);
y_limits6 = ylim;
plot([E_95_lower,E_95_lower], y_limits6, 'r--','LineWidth',1.5);
plot([E_95_upper,E_95_upper], y_limits6, 'r--','LineWidth',1.5);
x_limits7 = xlim;
plot(x_limits7, [0.025, 0.025], 'k:','LineWidth',1);
plot(x_limits7, [0.975, 0.975], 'k:','LineWidth',1);
xlabel('示值误差 (W)'); ylabel('累积概率'); title('示值误差累积分布函数(CDF)');
legend('CDF','95%包含区间','Location','southeast');
saveas(gcf, '5_累积分布函数.png','png');

figure(6); clf;
measurement_data = [individual_measurements, P_avg_simulated];
boxplot(measurement_data,'Labels',{'测量1','测量2','测量3','平均值'},'Colors','b');
grid on; ylabel('功率 (W)'); title('单次测量与平均值箱线图');
saveas(gcf, '6_测量值箱线图.png','png');

figure(7); clf;
normplot(E_simulated); grid on;
title('示值误差正态概率图');
saveas(gcf, '7_正态概率图.png','png');

figure(8); clf; set(gcf,'Position',[400,100,1000,600]);
c1 = [0.2,0.6,0.8]; c2 = [0.8,0.4,0.2]; c3 = [0.4,0.8,0.2];

subplot(1,2,1);
x_nums = 1:length(u_components); 
bar(x_nums, u_components); grid on;
set(gca, 'XTick', x_nums);       
set(gca, 'XTickLabel', u_names); 
patch_objs = findobj(gca,'Type','patch');
if length(patch_objs)>=3
    set(patch_objs(1),'FaceColor',c1,'EdgeColor','k');
    set(patch_objs(2),'FaceColor',c2,'EdgeColor','k');
    set(patch_objs(3),'FaceColor',c3,'EdgeColor','k');
end
for i=1:length(x_nums)
    text(x_nums(i), u_components(i)+0.1, sprintf('%.4f',u_components(i)), ...
         'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
end
xlabel('不确定度分量'); ylabel('标准不确定度 (W)'); title('不确定度分量绝对贡献');
ylim([0, max(u_components)*1.3]);

subplot(1,2,2);
pie(u_contrib);
colormap([c1;c2;c3]);
legend({sprintf('重复性 (%.2f%%)',contrib1),...
        sprintf('分辨力 (%.2f%%)',contrib2),...
        sprintf('MPE (%.2f%%)',contrib3)},...
        'Location','best','FontSize',10);
title('不确定度分量相对贡献（方差占比/计量规范）');

sgtitle('不确定度分量贡献分析','FontSize',14,'FontWeight','bold');
saveas(gcf, '8_不确定度分量贡献图.png','png');

fprintf('\n--- 模拟验证 ---\n');
fprintf('单次测量标准差：理论%.3f W | 模拟%.3f W\n',s,std(individual_measurements(:,1)));
fprintf('平均值标准差：理论%.3f W | 模拟%.3f W\n',s/sqrt(3),std(P_avg_simulated));
fprintf('理论与模拟相对差异：%.2f%%\n',abs(s/sqrt(3)-std(P_avg_simulated))/(s/sqrt(3))*100);

fprintf('\n--- 收敛性能 ---\n');
fprintf('初始(1000次)不确定度：%.4f W | 最终(10万次)：%.4f W\n',uncertainty_convergence(1),final_uncertainty);
fprintf('达到各阈值次数：');
for i=1:length(thresholds)
    fprintf('%.1f%%(%d次) ',thresholds(i),convergence_points_needed(i));
end
fprintf('\n\n所有8张图已全部生成并保存！\n');

fprintf('\n--- 10倍模拟次数节点的关键统计量 ---\n');
key_simulation_points = [1000, 10000, 100000, 1000000]; 

for idx_point = 1:length(key_simulation_points)
    target_n = key_simulation_points(idx_point);
    [~, closest_idx] = min(abs(simulation_counts - target_n));
    actual_n = simulation_counts(closest_idx);
    
    E_subset = E_simulated(1:actual_n);
    uncert_at_point = std(E_subset);          
    lower_95_at_point = prctile(E_subset, 2.5);  
    upper_95_at_point = prctile(E_subset, 97.5); 
    
    fprintf('模拟%8d次：标准不确定度=%.4f W | 95%%包含区间=[%.3f, %.3f] W\n', ...
            actual_n, uncert_at_point, lower_95_at_point, upper_95_at_point);
end

mean_seq = zeros(convergence_points, 1);       
std_seq = zeros(convergence_points, 1);        
lower95_seq = zeros(convergence_points, 1);    
upper95_seq = zeros(convergence_points, 1);    

for i = 1:convergence_points
    n_current = simulation_counts(i);          
    E_subset = E_simulated(1:n_current);       
    mean_seq(i) = mean(E_subset);              
    std_seq(i) = std(E_subset);                
    lower95_seq(i) = prctile(E_subset, 2.5);   
    upper95_seq(i) = prctile(E_subset, 97.5);  
end

tolerances_W = [1, 0.1, 0.01]; 
convergence_points_W = ones(size(tolerances_W)) * n_simulations;  
tols_crit1 = zeros(size(tolerances_W));  
tols_crit2 = zeros(size(tolerances_W));  
tols_crit3 = zeros(size(tolerances_W));  
tols_crit4 = zeros(size(tolerances_W));  
tols_uncert = zeros(size(tolerances_W)); 
tols_95lower = zeros(size(tolerances_W));
tols_95upper = zeros(size(tolerances_W));

fprintf('\n=== 容差收敛分析结果（四条件判定）===\n');
for tol_idx = 1:length(tolerances_W)
    tol_W = tolerances_W(tol_idx);
    is_converged = false;
    converged_idx = convergence_points;  
    
    for i = 2:convergence_points
        crit1 = 2 * std(mean_seq(1:i));       
        crit2 = 2 * std(std_seq(1:i));        
        crit3 = 2 * std(lower95_seq(1:i));    
        crit4 = 2 * std(upper95_seq(1:i));    
        
        if (crit1 < tol_W + 1e-8) && (crit2 < tol_W + 1e-8) && ...
           (crit3 < tol_W + 1e-8) && (crit4 < tol_W + 1e-8)
            is_converged = true;
            converged_idx = i;
            break;
        end
    end
    
    if is_converged
        convergence_points_W(tol_idx) = simulation_counts(converged_idx);
        tols_crit1(tol_idx) = 2 * std(mean_seq(1:converged_idx));
        tols_crit2(tol_idx) = 2 * std(std_seq(1:converged_idx));
        tols_crit3(tol_idx) = 2 * std(lower95_seq(1:converged_idx));
        tols_crit4(tol_idx) = 2 * std(upper95_seq(1:converged_idx));
        tols_uncert(tol_idx) = std_seq(converged_idx);
        tols_95lower(tol_idx) = lower95_seq(converged_idx);
        tols_95upper(tol_idx) = upper95_seq(converged_idx);
        
        fprintf('绝对容差 %.3fW: 达到收敛所需次数 = %d次\n', tol_W, convergence_points_W(tol_idx));
        fprintf('  ├─ 平均值的2倍标准差: %.6f W < %.3f W\n', tols_crit1(tol_idx), tol_W);
        fprintf('  ├─ 标准差的2倍标准差: %.6f W < %.3f W\n', tols_crit2(tol_idx), tol_W);
        fprintf('  ├─ 95%%左端点的2倍标准差: %.6f W < %.3f W\n', tols_crit3(tol_idx), tol_W);
        fprintf('  └─ 95%%右端点的2倍标准差: %.6f W < %.3f W\n', tols_crit4(tol_idx), tol_W);
    else
        tols_crit1(tol_idx) = NaN;
        tols_crit2(tol_idx) = NaN;
        tols_crit3(tol_idx) = NaN;
        tols_crit4(tol_idx) = NaN;
        tols_uncert(tol_idx) = NaN;
        tols_95lower(tol_idx) = NaN;
        tols_95upper(tol_idx) = NaN;
        fprintf('绝对容差 %.3fW: 未在%d次模拟内达到收敛（四条件未同时满足）\n', tol_W, n_simulations);
    end
end

figure(100); clf;
set(gcf, 'Position', [100, 100, 900, 700]);

valid_mask = convergence_points_W < n_simulations;
valid_tolerances = tolerances_W(valid_mask);
valid_points = convergence_points_W(valid_mask);

if ~isempty(valid_tolerances)
    bar_handle = bar(valid_tolerances, valid_points, 'FaceColor', [0.3, 0.6, 0.9], ...
                    'EdgeColor', 'k', 'LineWidth', 1.5, 'BarWidth', 0.6);
    grid on; 
    
    xlabel('绝对容差水平 (W)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('所需模拟次数（四条件达标）', 'FontSize', 12, 'FontWeight', 'bold');
    title('容差收敛速度柱状图（四条件判定：均值/标准差/95%区间均稳定）', 'FontSize', 14, 'FontWeight', 'bold');
    
    for j = 1:length(valid_points)
        text(valid_tolerances(j), valid_points(j) * 1.02, sprintf('%d', valid_points(j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontWeight', 'bold', 'FontSize', 11, 'Color', 'red');
    end
    
    if length(valid_tolerances) > 1
        set(gca, 'XScale', 'log');
        sorted_tols = sort(valid_tolerances);
        set(gca, 'XTick', sorted_tols);
        set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.3f', x), sorted_tols, 'UniformOutput', false));
    else
        set(gca, 'XTick', valid_tolerances);
        set(gca, 'XTickLabel', sprintf('%.3f', valid_tolerances));
    end
    
    grid on;
    set(gca, 'GridAlpha', 0.3);
else
    text(0.5, 0.5, '无满足四条件的收敛数据（容差要求过高）', ...
         'Units', 'normalized', 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'FontWeight', 'bold');
    xlabel('绝对容差水平 (W)');
    ylabel('所需模拟次数（四条件达标）');
    title('容差收敛速度柱状图（四条件判定）');
end

set(gca, 'Color', [0.95, 0.95, 0.95]);
saveas(gcf, '容差收敛速度柱状图_四条件判定.png', 'png');
fprintf('已保存: 容差收敛速度柱状图_四条件判定.png\n');

figure(101); clf;
set(gcf, 'Position', [150, 150, 900, 600]);

x_full = simulation_counts; 
crit1_seq = zeros(convergence_points, 1);
crit2_seq = zeros(convergence_points, 1);
crit3_seq = zeros(convergence_points, 1);
crit4_seq = zeros(convergence_points, 1);
for i = 2:convergence_points
    crit1_seq(i) = 2 * std(mean_seq(1:i));
    crit2_seq(i) = 2 * std(std_seq(1:i));
    crit3_seq(i) = 2 * std(lower95_seq(1:i));
    crit4_seq(i) = 2 * std(upper95_seq(1:i));
end

if ~isempty(x_full) && ~isempty(crit1_seq)
    h_crit1 = loglog(x_full, crit1_seq, 'r-', 'LineWidth', 2, 'DisplayName', '平均值的2倍标准差');
    hold on;
    h_crit2 = loglog(x_full, crit2_seq, 'g-', 'LineWidth', 2, 'DisplayName', '标准差的2倍标准差');
    h_crit3 = loglog(x_full, crit3_seq, 'b-', 'LineWidth', 2, 'DisplayName', '95%左端点的2倍标准差');
    h_crit4 = loglog(x_full, crit4_seq, 'm-', 'LineWidth', 2, 'DisplayName', '95%右端点的2倍标准差');
    grid on; grid minor;
    
    h_tol1 = plot([min(x_full), max(x_full)], [1, 1], 'k-', 'LineWidth', 2.5, 'DisplayName', '1W容差线');
    h_tol01 = plot([min(x_full), max(x_full)], [0.1, 0.1], 'k--', 'LineWidth', 2.5, 'DisplayName', '0.1W容差线');
    
    colors_tol = [[0.1,0.1,0.1]; [0.6,0.6,0.6]];
    for i = 1:length(tolerances_W)
        if valid_mask(i)
            point_idx = find(abs(simulation_counts - convergence_points_W(i)) < 1e-8, 1);
            if ~isempty(point_idx)
                plot(convergence_points_W(i), tolerances_W(i), 'ko', 'MarkerSize', 12, ...
                     'MarkerFaceColor', colors_tol(i,:), 'MarkerEdgeColor', 'k', 'LineWidth', 2);
                text(convergence_points_W(i)*1.1, tolerances_W(i)*0.8, ...
                     sprintf('@%d次', convergence_points_W(i)), ...
                     'FontSize', 10, 'Color', 'blue', 'FontWeight', 'bold');
            end
        end
    end
    
    xlim([min(x_full)*0.9, max(x_full)*1.1]); 
    xtick_vals = [100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]; 
    xtick_labels = {'10^2','5×10^2','10^3','5×10^3','10^4','5×10^4','10^5','5×10^5','10^6'};
    set(gca, 'XTick', xtick_vals);
    set(gca, 'XTickLabel', xtick_labels);
    
    all_crit = [crit1_seq(:); crit2_seq(:); crit3_seq(:); crit4_seq(:); tolerances_W(:)];
    y_min = min(all_crit(all_crit>0)) * 0.7;
    y_max = max(all_crit) * 2;
    ylim([y_min, y_max]);
    set(gca, 'GridAlpha', 0.3);
    set(gca, 'MinorGridAlpha', 0.1);
    
    legend([h_crit1,h_crit2,h_crit3,h_crit4,h_tol1,h_tol01], 'Location', 'best', 'FontSize', 9, 'NumColumns', 2);
else
    text(0.5, 0.5, '无收敛分析数据', ...
         'Units', 'normalized', 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'FontWeight', 'bold');
end

xlabel('模拟次数（对数尺度）', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('2倍标准差 / 容差水平 (W)', 'FontSize', 12, 'FontWeight', 'bold');
title('收敛速度分析（1W/0.1W+四条件判定）', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'Color', [0.95, 0.95, 0.95]);
saveas(gcf, '收敛速度分析对数图_1W01W_四条件.png', 'png');
fprintf('已保存: 收敛速度分析对数图_1W01W_四条件.png\n');

figure(103); clf;
set(gcf, 'Position', [250, 250, 1000, 700]);
axis off;

summary_text = cell(50,1); 
idx = 1;
summary_text{idx} = '蒙特卡洛模拟容差收敛性能分析总结（四条件判定）'; idx = idx+1;
summary_text{idx} = '判定规则：平均值/标准差/95%左端点/95%右端点的2倍标准差均<容差'; idx = idx+1;
summary_text{idx} = ''; idx = idx+1;
summary_text{idx} = '=== 容差收敛详细结果 ==='; idx = idx+1;

for i = 1:length(tolerances_W)
    summary_text{idx} = sprintf('  %.3fW容差:', tolerances_W(i)); idx = idx+1;
    if valid_mask(i)
        summary_text{idx} = sprintf('    收敛次数: %d次', convergence_points_W(i)); idx = idx+1;
        summary_text{idx} = sprintf('    平均值2倍标准差: %.6f W', tols_crit1(i)); idx = idx+1;
        summary_text{idx} = sprintf('    标准差2倍标准差: %.6f W', tols_crit2(i)); idx = idx+1;
        summary_text{idx} = sprintf('    95%%左端点2倍标准差: %.6f W', tols_crit3(i)); idx = idx+1;
        summary_text{idx} = sprintf('    95%%右端点2倍标准差: %.6f W', tols_crit4(i)); idx = idx+1;
        summary_text{idx} = sprintf('    达标点不确定度: %.4f W', tols_uncert(i)); idx = idx+1;
    else
        summary_text{idx} = sprintf('    状态: 未在%d次模拟内达标', n_simulations); idx = idx+1;
    end
    idx = idx+1;
end

summary_text{idx} = ''; idx = idx+1;
summary_text{idx} = '=== 全局最终指标 ==='; idx = idx+1;
summary_text{idx} = sprintf('最终标准不确定度: %.4f W', final_uncertainty); idx = idx+1;
summary_text{idx} = sprintf('最终95%%包含区间: [%.3f, %.3f] W', E_95_lower, E_95_upper); idx = idx+1;
summary_text{idx} = sprintf('总模拟次数: %d', n_simulations); idx = idx+1;

summary_text = summary_text(~cellfun(@isempty, summary_text));
text(0.05, 0.95, summary_text, 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'FontSize', 10, ...
     'FontName', 'FixedWidth', 'FontWeight', 'bold', ...
     'BackgroundColor', [0.9, 0.95, 1.0], ...
     'EdgeColor', 'blue', 'LineWidth', 2, ...
     'Margin', 15);

title('容差收敛性能总结（四条件判定）', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'blue');
saveas(gcf, '收敛性能总结_四条件判定.png', 'png');
fprintf('已保存: 收敛性能总结_四条件判定.png\n');

fprintf('\n═══════════════════════════════════════════════════\n');
fprintf('           四条件容差判定 - 详细收敛结果\n');
fprintf('═══════════════════════════════════════════════════\n');
fprintf('容差水平 (W)   所需模拟次数    状态        \n');
fprintf('-----------------------------------------------\n');

for i = 1:length(tolerances_W)
    achieved = '? 已达成';
    if ~valid_mask(i)
        achieved = '? 未达成';
    end
    fprintf('   %.3f         %8d       %s\n', tolerances_W(i), convergence_points_W(i), achieved);
end

fprintf('\n=== 四大判定指标详情 ===\n');
for i = 1:length(tolerances_W)
    if valid_mask(i)
        fprintf('达到%.3fW容差（%d次）：\n', tolerances_W(i), convergence_points_W(i));
        fprintf('  - 平均值的2倍标准差: %.6f W\n', tols_crit1(i));
        fprintf('  - 标准差的2倍标准差: %.6f W\n', tols_crit2(i));
        fprintf('  - 95%%左端点的2倍标准差: %.6f W\n', tols_crit3(i));
        fprintf('  - 95%%右端点的2倍标准差: %.6f W\n', tols_crit4(i));
        fprintf('  - 达标点95%%包含区间: [%.3f, %.3f] W\n', tols_95lower(i), tols_95upper(i));
    end
end

fprintf('\n=== 绘图完成 ===\n');
fprintf('已生成4张独立的容差收敛分析图（四条件判定）：\n');
fprintf('  1. 容差收敛速度柱状图_四条件判定.png\n');
fprintf('  2. 收敛速度分析对数图_四条件判定.png\n');
fprintf('  3. 所有容差收敛对比_四条件判定.png\n');
fprintf('  4. 收敛性能总结_四条件判定.png\n');

fprintf('\n═══════════════════════════════════════════════════\n');
fprintf('               四条件容差收敛性能分析报告\n');
fprintf('═══════════════════════════════════════════════════\n');

if any(tolerances_W==0.1)
    idx_01W = find(tolerances_W==0.1,1);
    if valid_mask(idx_01W)
        efficiency_01W = (final_uncertainty / 0.1) * (n_simulations / convergence_points_W(idx_01W));
        fprintf('0.1W容差收敛效率指标: %.3f（数值越小收敛越快）\n', efficiency_01W);
    end
end

if sum(valid_mask) > 0
    avg_convergence_speed = mean(convergence_points_W(valid_mask));
    fprintf('达标容差的平均收敛速度: %.0f次模拟/容差水平\n', avg_convergence_speed);
end

fprintf('报告生成时间: %s\n', datestr(now));
fprintf('═══════════════════════════════════════════════════\n');

fprintf('报告生成时间: %s\n', datestr(now));
fprintf('═══════════════════════════════════════════════════\n');