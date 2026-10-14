# Code for Publication

[简体中文](README.md) | **English**

This repository collects simulation code for research and metrology projects. Each project has its own folder. Select a project heading to open its folder, or select a source file to view it directly.

## Project folders

### [NO1 High Frequency Electrosurgical Units](https://github.com/TangXun1990/code-for-publication/tree/main/NO1%20High%20Frequency%20Electrosurgical%20Units)

MATLAB R2018b simulation of uncertainty in the indicated output-power error of high-frequency electrosurgical units.

Main source: [MCM](https://github.com/TangXun1990/code-for-publication/blob/main/NO1%20High%20Frequency%20Electrosurgical%20Units/HighFrequencyElectrosurgicalUnits202601.m)

### [NO2 Linear Accelerator](https://github.com/TangXun1990/code-for-publication/tree/main/NO2%20Linear%20Accelerator)

MATLAB R2024b simulation of uncertainty in absorbed dose to water from a linear accelerator.

Main sources: [GUM method](https://github.com/TangXun1990/code-for-publication/blob/main/NO2%20Linear%20Accelerator/NO2linearacceleratorGUM.m) · [MCM method](https://github.com/TangXun1990/code-for-publication/blob/main/NO2%20Linear%20Accelerator/NO2linearacceleratorMCM.m)

### [NO3 Ventilator Tidal Volume](https://github.com/TangXun1990/code-for-publication/tree/main/NO3%20Ventilator%20Tidal%20Volume)

Python 3.12.9 simulation of uncertainty in the relative indication error of ventilator tidal volume.

Main sources: [Correlation analysis](https://github.com/TangXun1990/code-for-publication/blob/main/NO3%20Ventilator%20Tidal%20Volume/Correlation.py) · [MCM method](https://github.com/TangXun1990/code-for-publication/blob/main/NO3%20Ventilator%20Tidal%20Volume/MCM.py)

## Project dependencies

### NO1 High Frequency Electrosurgical Units

- MATLAB R2018b.
- Statistics and Machine Learning Toolbox: the source calls `normpdf` and `prctile` to calculate the normal probability density and percentiles.

### NO2 Linear Accelerator

- MATLAB R2024b.
- The GUM script uses built-in MATLAB numerical, table and CSV functions; no additional toolbox dependency was identified.
- The MCM script requires Statistics and Machine Learning Toolbox (`normrnd`, `unifrnd`, `normpdf`) and Parallel Computing Toolbox (`gcp`, `parpool`, `parfor`).

### NO3 Ventilator Tidal Volume

- Python 3.12.9.
- `Correlation.py`: NumPy, Matplotlib, pandas and tqdm.
- `MCM.py`: NumPy, Matplotlib, SciPy (`scipy.stats.norm`) and Numba.
- Versions recorded in the repository: NumPy 2.4.2, SciPy 1.17.1, Matplotlib 3.10.8 and Numba 0.64.0. Versions for pandas and tqdm are not specified in the repository.
- Chinese text in plots also depends on Chinese fonts configured by the scripts being installed on the system; fonts are not Python package dependencies.

---

Source files are organized in their respective project folders. Open a project folder to browse or download its files.
