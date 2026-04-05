import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm

# ===================== 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ===================== 核心参数（全部改为双精度）=====================
n_sim = 10**7          # 测试用1e6，正式运行可改回10^7
n_measure = 3
bin_width = 0.05
r_min = -1.0
r_max = 1.0
r_step = 0.05
n_r_points = int((r_max - r_min) / r_step) + 1
r_values = np.linspace(r_min, r_max, n_r_points, dtype=np.float64)  # 双精度

# V0/Vm 真值（双精度）
V0_true = np.float64(620)
Vm_true = np.float64(574)

# V0 不确定度分量（单次测量，双精度）
V0_repeat_std_single = np.float64(5.50)  # 重复性单次标准差
V0_res_half = np.float64(0.5)            # 分辨力半宽（均匀分布）
V0_repeat_std_mean = V0_repeat_std_single / np.sqrt(np.float64(n_measure))  # 平均值标准差

# Vm 不确定度分量（单次测量，双精度）
Vm_repeat_std_single = np.float64(1.89)  # 重复性单次标准差
Vm_MPE_half = np.float64(600 * 0.03)     # MPE半宽（均匀分布）
Vm_res_half = np.float64(0.5)            # 分辨力半宽（均匀分布）
Vm_repeat_std_mean = Vm_repeat_std_single / np.sqrt(np.float64(n_measure))  # 平均值标准差

# 计算V0/Vm总合成标准不确定度（双精度）
u_V0 = np.sqrt(np.float64(V0_repeat_std_mean**2 + (V0_res_half/np.sqrt(np.float64(3)))**2))
u_Vm = np.sqrt(np.float64(Vm_repeat_std_mean**2 + (Vm_MPE_half/np.sqrt(np.float64(3)))**2 + (Vm_res_half/np.sqrt(np.float64(3)))**2))

# ========== Z轴手动配置（双精度） ==========
z_step_manual = np.float64(20000)
z_max_manual = np.float64(120000)
z_min_manual = np.float64(0)

# ===================== 模拟计算（全程双精度）=====================
std_uncertainties = []  # 存储每个r对应的标准不确定度
delta_all = []          # 存储每个r对应的δ数组

print(f"开始遍历相关系数（共{len(r_values)}个点）...")
for r in tqdm(r_values, desc="模拟进度", unit="个", ncols=80):
    # 生成严格相关的随机误差（双精度，避免正定矩阵报错）
    a = np.random.normal(0, 1, n_sim).astype(np.float64)  # 双精度随机数
    b = r * a + np.sqrt(np.float64(1 - r**2 + 1e-12)) * np.random.normal(0, 1, n_sim).astype(np.float64)
    
    # 生成V0和Vm的最终值（双精度）
    V0 = V0_true + u_V0 * a
    Vm = Vm_true + u_Vm * b
    
    # 计算相对示值误差δ（防止除零，双精度）
    Vm = np.clip(Vm, np.float64(1e-6), None)
    delta = (V0 / Vm - np.float64(1)) * np.float64(100)
    
    # 统计标准不确定度并存储（无偏标准差，双精度）
    std_uncertainty = np.std(delta, ddof=1, dtype=np.float64)
    std_uncertainties.append(std_uncertainty)
    delta_all.append(delta)

# 转换为数组方便计算（双精度）
std_uncertainties = np.array(std_uncertainties, dtype=np.float64)

# ===================== 统一频次分布的bins（双精度）=====================
delta_min = np.min(np.array([np.min(d) for d in delta_all], dtype=np.float64)) - np.float64(0.2)
delta_max = np.max(np.array([np.max(d) for d in delta_all], dtype=np.float64)) + np.float64(0.2)
bins = np.arange(delta_min, delta_max + bin_width, bin_width, dtype=np.float64)  # 双精度bins
bin_centers = (bins[:-1] + bins[1:]) / np.float64(2)  # 双精度bin中心

# 计算每个r对应的频次（双精度）
z_data = []
for delta in tqdm(delta_all, desc="频次计算", unit="条", ncols=80):
    counts, _ = np.histogram(delta, bins=bins, density=False)
    z_data.append(counts.astype(np.float64))  # 频次也转双精度
z_data = np.array(z_data, dtype=np.float64)

# ===================== 绘图1：标准不确定度随r变化 =====================
fig1 = plt.figure(figsize=(10, 6))
plt.plot(r_values, std_uncertainties, 'b-', linewidth=2.5, marker='o', markersize=4)
plt.grid(alpha=0.3)
plt.xlabel('相关系数 r')
plt.ylabel('标准不确定度 u_c (%)')
plt.title('标准不确定度随相关系数的变化趋势')
plt.xlim(r_min - np.float64(0.05), r_max + np.float64(0.05))
y_min = np.min(std_uncertainties) - np.float64(0.02)
y_max = np.max(std_uncertainties) + np.float64(0.02)
plt.ylim(y_min, y_max)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
plt.tight_layout()
fig1.savefig('标准不确定度-相关系数变化图.svg', format='svg')
plt.close(fig1)
print("✅ 图1（标准不确定度变化图）已导出")

# ===================== 绘图2：3D频次曲线（无95%红点）=====================
fig2 = plt.figure(figsize=(12, 8))
ax = fig2.add_subplot(111, projection='3d')

# 仅绘制纯净的频次曲线，无任何红点
for i, r in enumerate(r_values):
    ax.plot(bin_centers,
            np.full_like(bin_centers, r, dtype=np.float64),  # 双精度填充
            z_data[i],
            color='#00BFFF',
            linewidth=1.2,
            alpha=0.8)

# 手动配置坐标轴（双精度）
ax.set_yticks(np.arange(r_min, r_max + np.float64(0.2), np.float64(0.2), dtype=np.float64))
zticks_manual = np.arange(z_min_manual, z_max_manual + z_step_manual, z_step_manual, dtype=np.float64)
ax.set_zticks(zticks_manual)
ax.set_zlim(z_min_manual, z_max_manual)

# X轴范围优化（双精度）
x_min = np.floor(delta_min / np.float64(2)) * np.float64(2)
x_max = np.ceil(delta_max / np.float64(2)) * np.float64(2)
ax.set_xticks(np.arange(x_min, x_max + np.float64(2), np.float64(2), dtype=np.float64))

# 坐标轴标签
ax.set_xlabel('相对示值误差 δ (%)', fontsize=10)
ax.set_ylabel('相关系数 r', fontsize=10)
ax.set_zlabel('实际频次', fontsize=10)
ax.set_title('各相关系数对应的示值误差频次曲线（无95%端点）', fontsize=12)
ax.view_init(elev=25, azim=60)

plt.tight_layout()
fig2.savefig('示值误差频次曲线-3D图_无红点.svg', format='svg')
plt.close(fig2)
print("✅ 图2（3D频次曲线，无红点）已导出")

# ===================== 保存完整的r和标准不确定度数据到Excel =====================
df_uncertainty = pd.DataFrame({
    '相关系数 r': r_values.round(4),
    '标准不确定度 u_c (%)': std_uncertainties.round(6)
})

# 保存频次数据（可选）
df_frequency = pd.DataFrame(
    data=np.column_stack([bin_centers.round(4), z_data.T]),
    columns=['示值误差 δ (%)'] + [f'r={r:.2f}' for r in r_values]
)
for col in df_frequency.columns[1:]:
    df_frequency[col] = df_frequency[col].astype(int)

# 写入Excel
excel_filename = '相关系数-标准不确定度完整数据.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df_uncertainty.to_excel(writer, sheet_name='标准不确定度数据', index=False)
    df_frequency.to_excel(writer, sheet_name='频次分布数据', index=False)

print(f"✅ 完整数据已保存到：{excel_filename}")

# ===================== 输出关键数值结果（重点保留r和标准不确定度）=====================
print("\n===== 关键统计结果（相关系数r & 标准不确定度） =====")
print(f"相关系数范围：{r_min} ~ {r_max}，步进{r_step}")
print(f"最小标准不确定度：{np.min(std_uncertainties):.4f}%（r={r_values[np.argmin(std_uncertainties)]:.2f}）")
print(f"最大标准不确定度：{np.max(std_uncertainties):.4f}%（r={r_values[np.argmax(std_uncertainties)]:.2f}）")

# 输出关键r值对应的标准不确定度
key_r_values = [-1.0, 0.0, 0.818, 1.0]
for key_r in key_r_values:
    idx = np.argmin(np.abs(r_values - np.float64(key_r)))
    print(f"r={key_r:.3f} 时，标准不确定度：{std_uncertainties[idx]:.4f}%")

# 输出所有r和对应的标准不确定度（可选，如需完整列表可取消注释）
# print("\n===== 所有相关系数对应的标准不确定度 =====")
# for r, uc in zip(r_values, std_uncertainties):
#     print(f"r={r:.2f} → u_c={uc:.4f}%")