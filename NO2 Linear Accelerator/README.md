# NO2 直线加速器 | Linear Accelerator

## 简体中文

### 项目依赖

- MATLAB R2024b。
- GUM 脚本使用 MATLAB 内置数值、表格和 CSV 函数，无额外工具箱依赖。
- MCM 脚本需要 Statistics and Machine Learning Toolbox（`normrnd`、`unifrnd`、`normpdf`）以及 Parallel Computing Toolbox（`gcp`、`parpool`、`parfor`）。

### 主要文件

- [NO2linearacceleratorGUM.m](NO2linearacceleratorGUM.m)：GUM 计算。
- [NO2linearacceleratorMCM.m](NO2linearacceleratorMCM.m)：蒙特卡洛计算。

## English

### Dependencies

- MATLAB R2024b.
- The GUM script uses built-in MATLAB numerical, table and CSV functions; no additional toolbox is required.
- The MCM script requires Statistics and Machine Learning Toolbox (`normrnd`, `unifrnd`, `normpdf`) and Parallel Computing Toolbox (`gcp`, `parpool`, `parfor`).

### Main files

- [NO2linearacceleratorGUM.m](NO2linearacceleratorGUM.m): GUM calculation.
- [NO2linearacceleratorMCM.m](NO2linearacceleratorMCM.m): Monte Carlo calculation.
