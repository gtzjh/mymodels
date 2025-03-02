# Log

## 20250302

调整模块结构：把回归模型模块移动到单独的文件模块中，并针对这一模块添加测试代码：tests/test_regressors.py

测试项目包括：

- 模型初始化测试：验证所有支持模型能否正确初始化
- 异常处理测试：检查无效输入时的错误处理
- 类别特征处理测试：验证支持类别特征的模型是否正确处理
- 参数空间验证：确保参数空间结构符合预期
- 静态参数验证：检查关键静态参数设置
- 边界条件测试：测试空类别特征列表等特殊情况
- 模型特定功能测试：验证不同模型的特殊处理逻辑

## 20250223

Completed support for categorical variables in models training and testing process.

## 20250218

- Implemented the encoder module to encode categorical features.
- Supported encoding methods: OneHot, Label, Target, Frequency, Binary, Ordinal.
- Added functionality to fit and transform data using the specified encoding method.
- Included validation to ensure the encoding process does not introduce null values or change the number of rows.
- Provided a method to retrieve the mapping relationships for each encoding method.
- Added a utility to convert numpy types to native Python types for JSON serialization.
- Implemented saving and loading of the encoder using joblib.
- Ensured compatibility with large datasets by issuing warnings for potential high memory usage.
- Added comprehensive error handling and validation checks for input data.

## 20250214

Fixed the bug of the reproducible optimization process.

Added the static parameters for the optimization process.

Added SVR, KNR, and MLP.

Adjusted the regressor structure to OOP.

## 20250213

Support the categorical features.

Adjusted the README.md.

Added XGBoost and AdaBoost.

Add the dependency plot for SHAP.

## 20250207

Corrected the partial dependency plot (PDP).

Simplified the code.

Abandon the gpu support, for 3 reasons:

- Data moving from memory to gpu's memory will take more time.

- The dataset is not big enough for gpu to perform better than cpu.

- Some contradition will occur when using gpu, and not every type of model has perfect support on gpu.

## 20241206

Remove the R plotting code for scatter plot.

## 20241129

Remove the R plotting code for SHAP summary plot and partial dependency plot (PDP).

## 2024.11.27
1. Add GBDT.

2. Used another dataset for demonstration.

## 2024.10.10

1. Add Chinese version.

## 2024.07.03

1. Add figures of results.

2. Change the stop of installing environment.

## 2024.07.02

1. Due to the initial parameters are randomly set, some warnings may occur when implement CatBoost. Just ignore it or repeat.

2. The categorical features supporting must be rewriten.

## 2024.06.30

1. Output the accuracy, best parameters, optimization, and shap results.

## 2024.06.15

1.  Remove Autogluon.

2.  Modify NN module.

3.  Tested on linux (Ubuntu 22.04, WSL2)

## 2024.06.09

1.  Added lightgbm

## 2024.06.09

1.  Found that the KernelExplainer can not calculate interaction values.

2.  Finished the tree based model, and finished the test.

3.  Plan to extract some same codes from each seperated .py file, and rebuild the whole `models` module.

4.  The optuna should accomadate the consistent parameters and pass it to the final best parameters set, but now they were all ignored.

*Created by [Junhong](https://github.com/gtzjh). All rights reserved.*