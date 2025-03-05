import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import yaml, pathlib, json


plt.rc('font', family = 'Times New Roman')


def _regr_accuracy_plot(_r2_value, _rmse_value, _mae_value, _y, _y_pred, _results_dir):
    plt.figure(figsize = (8, 8), dpi = 500)
    plt.scatter(_y, _y_pred, color = '#4682B4', alpha = 0.4, s = 150)
    
    # 定义x轴的上下限，为了防止轴须图中的离散值改变了坐标轴定位
    _min = _y.min() - abs(_y.min()) * 0.15
    _max = _y.max() + abs(_y.max()) * 0.15
    plt.xlim(_min, _max)
    plt.ylim(_min, _max)

    # 进行散点的线性拟合
    param = np.polyfit(_y, _y_pred, 1)
    y2 = param[0] * _y + param[1]

    _y = _y.to_numpy()
    y2 = y2.to_numpy()

    _line1, = plt.plot(_y, y2, color = 'black', label = 'y = ' + f'{param[0]:.2f}' + " * x" + " + " + f'{param[1]:.2f}')
    _line2, = plt.plot([_min, _max], [_min, _max], '--', color = 'gray', label = 'y = x') # 绘制 y = x 的虚线
    _plot_r2, = plt.plot(0, 0, '-', color = 'w', label = f'R2      :  {_r2_value:.3f}')
    _plot_rmse, = plt.plot(0, 0, '-', color = 'w', label = f'RMSE:  {_rmse_value:.3f}')
    _plot_mae, = plt.plot(0, 0, '-', color = 'w', label = f'MAE  :  {_mae_value:.3f}')

    plt.legend(handles = [_line1, _line2, _plot_r2, _plot_rmse, _plot_mae], 
               loc = 'upper left', fancybox = True, shadow = True, fontsize = 16, prop = {'size': 16})
    
    plt.ylabel('Predicted values', fontdict = {'size': 18})
    plt.xlabel('Actual values', fontdict = {'size': 18})
    plt.yticks(size = 16)
    plt.xticks(size = 16)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.savefig(_results_dir.joinpath('accuracy_plot.jpg'), dpi = 500)
    plt.close()

    return None


# 输出精度结果
def regr_accuracy(y_test, y_test_pred, y_train, y_train_pred, results_dir):
    ###########################################################################
    # Check variables' type
    assert isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series) or isinstance(y_test, np.ndarray)
    assert isinstance(y_test_pred, pd.DataFrame) or isinstance(y_test_pred, pd.Series) or isinstance(y_test_pred, np.ndarray)
    assert isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series) or isinstance(y_train, np.ndarray)
    assert isinstance(y_train_pred, pd.DataFrame) or isinstance(y_train_pred, pd.Series) or isinstance(y_train_pred, np.ndarray)
    assert isinstance(results_dir, str) or isinstance(results_dir, pathlib.Path)
    results_dir = pathlib.Path(results_dir)
    ###########################################################################

    ###########################################################################
    # Output train and test results
    test_results = pd.DataFrame(data = {"y_test": y_test,
                                        "y_test_pred": y_test_pred})
    train_results = pd.DataFrame(data = {"y_train": y_train,
                                         "y_train_pred": y_train_pred})
    ###########################################################################

    ###########################################################################
    # Accuracy
    accuracy_dict = dict({
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "test_rmse": float(root_mean_squared_error(y_test, y_test_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "train_rmse": float(root_mean_squared_error(y_train, y_train_pred)),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred))
    })
    with open(results_dir.joinpath("accuracy.yml"), 'w', encoding = "utf-8") as file:
        yaml.dump(accuracy_dict, file)
    print("Accuracy:")
    print(json.dumps(accuracy_dict, indent=4))
    ###########################################################################

    ###########################################################################
    # Plot
    _regr_accuracy_plot(
        accuracy_dict["test_r2"],
        accuracy_dict["test_rmse"],
        accuracy_dict["test_mae"],
        y_test,
        y_test_pred,
        results_dir
    )
    ###########################################################################

    return None


def evaluate(
    model_name: str,
    opt_model,
    x_test: pd.DataFrame,
    y_test,
    x_train: pd.DataFrame,
    y_train,
    results_dir: str | pathlib.Path,
    encoder = None,
) -> None:
    
    _final_x_test = x_test
    _final_x_train = x_train

    if model_name != "cat":
        if encoder is not None:
            _final_x_test = encoder.transform(X=x_test)
            _final_x_train = encoder.transform(X=x_train)

    
    # 评估并保存结果
    _y_test_pred = opt_model.predict(_final_x_test)    # 测试集上的准确度
    _y_train_pred = opt_model.predict(_final_x_train)  # 训练集上的准确度
    regr_accuracy(y_test, _y_test_pred, y_train, _y_train_pred, results_dir)
    
    return None


"""
if __name__ == "__main__":
    scatter_test = pd.read_csv("results/rf/scatter_test.csv", encoding = "utf-8")
    scatter_train = pd.read_csv("results/rf/scatter_train.csv", encoding = "utf-8")
    regr_accuracy(
        y_test = scatter_test["y_test"],
        y_test_pred = scatter_test["y_test_pred"],
        y_train = scatter_train["y_train"],
        y_train_pred = scatter_train["y_train_pred"],
        results_dir = "",
    )
"""
