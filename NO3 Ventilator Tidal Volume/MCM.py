import os
import sys
os.environ["NUMBA_NUM_THREADS"] = str(os.cpu_count())
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from numba import jit, prange, set_num_threads
import atexit

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open("MCM_result.log", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
atexit.register(log_file.close)

cpu_count = os.cpu_count()
set_num_threads(cpu_count)

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Source Han Sans CN', 'SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

SQRT_2PI = np.sqrt(2 * np.pi)

n_sim = 10**8
n_measure = 3
step_size = 1000
convergence_tolerances = [1.0, 0.5, 0.1, 0.05, 0.01]

V0_true = 620.0
Vm_true = 574.0

u_V0 = np.sqrt((5.50/np.sqrt(n_measure))**2 + (0.5/np.sqrt(3))**2)
u_Vm = np.sqrt((1.89/np.sqrt(n_measure))**2 + (600*0.03/np.sqrt(3))**2 + (0.5/np.sqrt(3))**2)
r = 0.418

@jit(nopython=True, fastmath=True, parallel=True)
def generate_correlated_mcmc(V0_true, Vm_true, u_V0, u_Vm, r, n_sim):
    delta = np.empty(n_sim, dtype=np.float64)
    for i in prange(n_sim):
        z1 = np.random.normal(0, 1)
        z2 = np.random.normal(0, 1)

        x_V0 = z1
        x_Vm = r * z1 + np.sqrt(1.0 - r**2) * z2

        err_V0 = u_V0 * x_V0
        err_Vm = u_Vm * x_Vm

        V0 = V0_true + err_V0
        Vm = Vm_true + err_Vm

        Vm = max(Vm, 1e-6)

        delta[i] = (V0 / Vm - 1.0) * 100.0
    return delta

@jit(nopython=True, fastmath=True, parallel=True)
def compute_convergence(delta, steps):
    n_steps = len(steps)
    conv_mean = np.empty(n_steps, dtype=np.float64)
    conv_uc   = np.empty(n_steps, dtype=np.float64)
    mean_2std = np.empty(n_steps, dtype=np.float64)
    std_2std  = np.empty(n_steps, dtype=np.float64)
    ci95_l    = np.empty(n_steps, dtype=np.float64)
    ci95_r    = np.empty(n_steps, dtype=np.float64)

    for idx in prange(n_steps):
        N = steps[idx]
        d = delta[:N]
        mu  = np.mean(d)
        sig = np.std(d)
        low  = np.percentile(d, 2.5)
        high = np.percentile(d, 97.5)

        conv_mean[idx] = mu
        conv_uc[idx]   = sig

        mean_2std[idx] = 2.0 * sig / np.sqrt(N)
        std_2std[idx]  = 2.0 * sig / np.sqrt(2.0*(N-1)) if N>1 else 1e9

        pdf_low = np.exp(-(low - mu)**2 / (2 * sig**2)) / (sig * SQRT_2PI)
        pdf_high = np.exp(-(high - mu)**2 / (2 * sig**2)) / (sig * SQRT_2PI)

        se_low = np.sqrt(0.025 * 0.975 / (N * pdf_low **2)) if pdf_low > 1e-12 else 1e9
        se_high = np.sqrt(0.975 * 0.025 / (N * pdf_high **2)) if pdf_high > 1e-12 else 1e9

        ci95_l[idx] = 2.0 * se_low
        ci95_r[idx] = 2.0 * se_high

    return conv_mean, conv_uc, mean_2std, std_2std, ci95_l, ci95_r

@jit(nopython=True)
def first_below(arr, tol, continuous_require=5):
    n = len(arr)
    current_continuous = 0
    for i in range(n):
        if arr[i] < tol:
            current_continuous += 1
            if current_continuous >= continuous_require:
                return i - continuous_require + 1
        else:
            current_continuous = 0
    return -1

if __name__ == '__main__':
    print("==================================================")
    print("  JJF1059.2 严格4条收敛线 | 左右完全独立不重叠 | 图1固定0.2%间隔")
    print("==================================================")
    t0 = time.time()

    delta = generate_correlated_mcmc(V0_true, Vm_true, u_V0, u_Vm, r, n_sim)
    steps = np.arange(step_size, n_sim+1, step_size, dtype=np.int64)
    conv_mean, conv_uc, mean_2std, std_2std, ci95_l, ci95_r = compute_convergence(delta, steps)

    test_N = 100000
    z1_test = np.random.normal(0,1,test_N)
    z2_test = np.random.normal(0,1,test_N)
    x_test = z1_test
    y_test = r*z1_test + np.sqrt(1-r**2)*z2_test
    real_r = np.corrcoef(x_test, y_test)[0,1]
    print(f"理论 r = {r:.4f} | 实际抽样 r = {real_r:.4f}")

    print("\n" + "="*90)
    print("  容差 | 均值2σ | 标准差2σ | 左端点2σ | 右端点2σ | 全部达标 N")
    print("="*90)

    for tol in convergence_tolerances:
        i_m = first_below(mean_2std, tol)
        i_s = first_below(std_2std, tol)
        i_l = first_below(ci95_l, tol)
        i_r = first_below(ci95_r, tol)
        valid = [i for i in [i_m,i_s,i_l,i_r] if i >= 0]
        if len(valid) <4:
            print(f"\n⚠️  容差 {tol:.2f}% 未全部达标")
            continue
        i_all = max(valid)
        N_all = steps[i_all]
        mu = conv_mean[i_all]
        uc = conv_uc[i_all]
        lo = np.percentile(delta[:N_all],2.5)
        hi = np.percentile(delta[:N_all],97.5)

        print(f"\n✅ 容差 {tol:.2f}%")
        print(f"  均值2σ:{steps[i_m]:>6}  标准差2σ:{steps[i_s]:>6}")
        print(f"  左端点:{steps[i_l]:>6}  右端点  :{steps[i_r]:>6}")
        print(f"  🎯 全部达标 N={N_all}")
        print(f"  均值={mu:.6f}%  uc={uc:.6f}%  95%=[{lo:.6f},{hi:.6f}]")

    md = np.mean(delta)
    sd = np.std(delta)
    final_low = np.percentile(delta,2.5)
    final_high= np.percentile(delta,97.5)

    plt.figure(figsize=(8,6))
    bin_width = 0.2
    bin_start = np.floor(delta.min() / bin_width) * bin_width
    bin_end = np.ceil(delta.max() / bin_width) * bin_width
    bins = np.arange(bin_start, bin_end + bin_width, bin_width)

    plt.hist(delta, bins=bins, color='lightblue', alpha=0.7, edgecolor='black')
    x = np.linspace(delta.min(), delta.max(),1000)
    y = norm.pdf(x, md, sd) * len(delta) * bin_width
    plt.plot(x,y,'k-',lw=2,label='正态拟合')
    plt.axvline(md,c='red',ls='--',lw=2,label=f'均值={md:.3f}%')
    plt.axvline(final_low,c='orange',ls=':')
    plt.axvline(final_high,c='orange',ls=':')
    plt.xlabel('相对误差 δ (%)')
    plt.ylabel('频次')
    plt.title(f'MCM分布 r≈{real_r:.3f} | 固定间隔=0.2%')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('MCM_dist.svg', format='svg')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(steps, conv_mean, 'b-', lw=1)
    plt.xscale('log')
    plt.xlabel('模拟次数')
    plt.ylabel('δ 均值')
    plt.title('均值收敛')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('MCM_mean.svg', format='svg')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(steps, conv_uc, 'g-', lw=1)
    plt.xscale('log')
    plt.xlabel('模拟次数')
    plt.ylabel('u_c')
    plt.title('不确定度收敛')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('MCM_uc.svg', format='svg')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(steps, mean_2std, 'b', label='均值2σ', linewidth=1.2)
    plt.plot(steps, std_2std,  'g', label='标准差2σ', linewidth=1.2)
    plt.plot(steps, ci95_l,    'r', label='95%左端点2σ', linewidth=1.2)
    plt.plot(steps, ci95_r,   'm', label='95%右端点2σ', linewidth=1.2)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(top=1.0)
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('MCM_conv.svg', format='svg')
    plt.close()

    print("\n✅ 运行完成 —— 图1固定0.2%间隔，图4四条线完全独立！")
    print(f"总耗时：{time.time()-t0:.2f} s")
    print(f"δ 均值 = {md:.6f}%")
    print(f"u_c    = {sd:.6f}%")
    print(f"95%区间 = [{final_low:.6f}, {final_high:.6f}]")