# 论文复现代码

**简体中文** | [English](README.en.md)

本仓库收录用于论文和计量研究的模拟代码，按项目分目录整理。点击项目标题可直接进入相应文件夹，点击源码名称可直接打开文件。

## 项目目录

### [NO1 高频电刀](https://github.com/TangXun1990/code-for-publication/tree/main/NO1%20High%20Frequency%20Electrosurgical%20Units)

使用 MATLAB R2018b 模拟高频电刀输出功率示值误差的不确定度。

主要源码：[MCM](https://github.com/TangXun1990/code-for-publication/blob/main/NO1%20High%20Frequency%20Electrosurgical%20Units/HighFrequencyElectrosurgicalUnits202601.m)

### [NO2 直线加速器](https://github.com/TangXun1990/code-for-publication/tree/main/NO2%20Linear%20Accelerator)

使用 MATLAB R2024b 模拟直线加速器水中吸收剂量的不确定度。

主要源码：[GUM 方法](https://github.com/TangXun1990/code-for-publication/blob/main/NO2%20Linear%20Accelerator/NO2linearacceleratorGUM.m) · [MCM 方法](https://github.com/TangXun1990/code-for-publication/blob/main/NO2%20Linear%20Accelerator/NO2linearacceleratorMCM.m)

### [NO3 呼吸机潮气量](https://github.com/TangXun1990/code-for-publication/tree/main/NO3%20Ventilator%20Tidal%20Volume)

使用 Python 3.12.9 模拟呼吸机潮气量相对示值误差的不确定度。

主要源码：[相关性分析](https://github.com/TangXun1990/code-for-publication/blob/main/NO3%20Ventilator%20Tidal%20Volume/Correlation.py) · [MCM 方法](https://github.com/TangXun1990/code-for-publication/blob/main/NO3%20Ventilator%20Tidal%20Volume/MCM.py)

## 呼吸机项目依赖

- NumPy 2.4.2
- SciPy 1.17.1
- Matplotlib 3.10.8
- Numba 0.64.0

以上版本按仓库原 README 记录；运行前请根据本机 Python 环境确认相关库可用。

---

各项目的源码和文件均保存在对应目录中。欢迎通过项目文件夹查看或下载。
