import numpy as np
import pandas as pd
import matplotlib
# 设置后端为Agg，这是一个非交互式后端，避免线程相关问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)



def _plot_regression_scatter(_y, _y_pred):
    """Creates a scatter plot of actual vs predicted values for regression model evaluation.

    This function generates a visualization comparing actual values to predicted values
    from a regression model. It includes a scatter plot of the data points, a linear fit line,
    a y=x reference line, and displays performance metrics (R2, RMSE, MAE) in the legend.

    Args:
        _y : Actual target values.
        _y_pred : Predicted target values.

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects that can be further customized or saved.
    """

    if hasattr(_y, 'ndim'):
        assert (_y.ndim == 1), "Input _y must be a 1D array-like structure."
    else:
        assert isinstance(_y, (list, np.ndarray, pd.Series)), "Input _y must be a 1D array-like structure."

    if hasattr(_y_pred, 'ndim'):
        assert (_y_pred.ndim == 1), "Input _y_pred must be a 1D array-like structure."
    else:
        assert isinstance(_y_pred, (list, np.ndarray, pd.Series)), "Input _y_pred must be a 1D array-like structure."
    
    assert np.issubdtype(np.array(_y).dtype, np.number), "All values in _y must be numeric."
    assert np.issubdtype(np.array(_y_pred).dtype, np.number), "All values in _y_pred must be numeric."


    assert len(_y) == len(_y_pred), "The length of _y and _y_pred must be the same."

    assert not np.any(pd.isnull(_y)), "Input _y must not contain any empty values."
    assert not np.any(pd.isnull(_y_pred)), "Input _y_pred must not contain any empty values."


    fig = plt.figure(figsize = (8, 8), dpi = 500)
    ax = fig.add_subplot(111)
    ax.scatter(_y, _y_pred, color = '#4682B4', alpha = 0.4, s = 150)
    
    # 定义x轴的上下限，为了防止轴须图中的离散值改变了坐标轴定位
    _min = _y.min() - abs(_y.min()) * 0.15
    _max = _y.max() + abs(_y.max()) * 0.15
    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)

    # 进行散点的线性拟合
    param = np.polyfit(_y, _y_pred, 1)
    y2 = param[0] * _y + param[1]


    _line1, = ax.plot(_y, y2, color = 'black', label = 'y = ' + f'{param[0]:.2f}' + " * x" + " + " + f'{param[1]:.2f}')
    _line2, = ax.plot([_min, _max], [_min, _max], '--', color = 'gray', label = 'y = x') # 绘制 y = x 的虚线

    ax.legend(handles = [_line1, _line2], 
            loc = 'upper left', fancybox = True, shadow = True, fontsize = 16, prop = {'size': 16})
    
    ax.set_ylabel('Predicted values', fontdict = {'size': 18})
    ax.set_xlabel('Actual values', fontdict = {'size': 18})
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    return fig, ax
