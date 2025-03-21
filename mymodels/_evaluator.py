import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import yaml, pathlib, json


plt.rc('font', family = 'Times New Roman')


def _plot_classifier(y_test, y_test_pred, y_train, y_train_pred, results_dir):
    pass
    return None


def _plot_regr(_r2_value, _rmse_value, _mae_value, _y, _y_pred, _results_dir):
    """Creates a scatter plot of actual vs predicted values for regression model evaluation.

    This function generates a visualization comparing actual values to predicted values
    from a regression model. It includes a scatter plot of the data points, a linear fit line,
    a y=x reference line, and displays performance metrics (R2, RMSE, MAE) in the legend.

    Args:
        _r2_value (float): R-squared value of the regression model.
        _rmse_value (float): Root Mean Squared Error value of the regression model.
        _mae_value (float): Mean Absolute Error value of the regression model.
        _y (pandas.Series): Actual target values.
        _y_pred (pandas.Series or numpy.ndarray): Predicted target values.
        _results_dir (str or pathlib.Path): Directory where the plot image will be saved.

    Returns:
        None: The function saves the plot to disk and does not return a value.
    """
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



class Evaluator:
    """A class for evaluating machine learning regression models.
    
    This class handles the evaluation of machine learning models, computing various 
    accuracy metrics (R², RMSE, MAE), visualizing actual vs predicted values, 
    and saving/printing results.
    """
    def __init__(
            self,
            model_name,
            results_dir: str | pathlib.Path,
            plot = True,   # Plot scatter plot
            print_results = True,  # Print results to console
            save_results = True,    # Save results to files
            save_raw_data = False    # Save raw data to files
        ):
        """Initialize the Evaluator with model and configuration settings.
        
        Args:
            results_dir (str | pathlib.Path): Directory to save evaluation results.
            plot (bool, optional): Whether to generate accuracy plots. Defaults to True.
            print_results (bool, optional): Whether to print results to console. Defaults to True.
            save_results (bool, optional): Whether to save results to files. Defaults to True.
            save_raw_data (bool, optional): Whether to save raw prediction data. Defaults to False.
        """
        self.model_name = model_name
        self.results_dir = pathlib.Path(results_dir)

        # Options
        self.plot = plot
        self.print_results = print_results
        self.save_results = save_results
        self.save_raw_data = save_raw_data

        # Will change in runtime
        self.accuracy_dict = None

    
    def evaluate(self, y_test, y_test_pred, y_train, y_train_pred):
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        self.y_train = y_train
        self.y_train_pred = y_train_pred
        """Evaluate the model on test and training data.
        
        Performs model evaluation by:
        1. Computing accuracy metrics (R2, RMSE, MAE for regression; accuracy, precision, recall, F1 for classification)
        2. Processing output options (saving results to files, printing to console, plotting visualizations)
        
        Args:
            y_test: Actual test target values
            y_test_pred: Predicted test target values
            y_train: Actual training target values
            y_train_pred: Predicted training target values
        """

        if self.model_name in ["svr", "knr", "mlpr", "adar", \
                               "dtr", "rfr", "gbdtr", "xgbr", "lgbr", "catr"]:
            self._get_accuracy_4_regression_task(
                self.y_test, self.y_test_pred, self.y_train, self.y_train_pred
            )
        else:
            self._get_accuracy_4_classification_task(
                self.y_test, self.y_test_pred, self.y_train, self.y_train_pred
            )

        # Save results
        self._output()

        return None


    def _get_accuracy_4_regression_task(self, y_test, y_test_pred, y_train, y_train_pred):
        """Calculate regression accuracy metrics for both test and training data.
        
        Computes R², RMSE, and MAE metrics for both test and training predictions
        and stores them in the accuracy_dict attribute.
        
        Args:
            y_test: Actual test target values.
            y_test_pred: Predicted test target values.
            y_train: Actual training target values.
            y_train_pred: Predicted training target values.
        """
        self.accuracy_dict = dict({
            "test_r2": float(r2_score(y_test, y_test_pred)),
            "test_rmse": float(root_mean_squared_error(y_test, y_test_pred)),
            "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "train_rmse": float(root_mean_squared_error(y_train, y_train_pred)),
            "train_mae": float(mean_absolute_error(y_train, y_train_pred))
        })
        return None
    

    def _get_accuracy_4_classification_task(self, y_test, y_test_pred, y_train, y_train_pred):
        """Calculate classification accuracy metrics for both test and training data.
        
        Computes accuracy, precision, recall, and F1 metrics for both test and training predictions
        and stores them in the accuracy_dict attribute.
        
        Args:
            y_test: Actual test target values.
            y_test_pred: Predicted test target values.
            y_train: Actual training target values.
            y_train_pred: Predicted training target values.
        """
        self.accuracy_dict = dict({
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "test_precision": float(precision_score(y_test, y_test_pred, average = "weighted")),
            "test_recall": float(recall_score(y_test, y_test_pred, average = "weighted")),
            "test_f1": float(f1_score(y_test, y_test_pred, average = "weighted")),
            "test_kappa": float(cohen_kappa_score(y_test, y_test_pred)),
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "train_precision": float(precision_score(y_train, y_train_pred, average = "weighted")),
            "train_recall": float(recall_score(y_train, y_train_pred, average = "weighted")),
            "train_f1": float(f1_score(y_train, y_train_pred, average = "weighted")),
            "train_kappa": float(cohen_kappa_score(y_train, y_train_pred))
        })
        return None


    def _output(self):
        """Process output options based on configuration.
        
        Handles saving results to files, printing to console, generating plots,
        and saving raw prediction data based on the configuration settings.
        """
        # Save results to files
        if self.save_results:
            with open(self.results_dir.joinpath("accuracy.yml"), 'w', encoding = "utf-8") as file:
                yaml.dump(self.accuracy_dict, file)
        
        # Print results to the console
        if self.print_results:
            print(f"Accuracy of {self.model_name}: \n", \
                  json.dumps(self.accuracy_dict, indent=4))

        # Plot
        if self.plot:
            _plot_regr(
                self.accuracy_dict["test_r2"],
                self.accuracy_dict["test_rmse"],
                self.accuracy_dict["test_mae"],
                self.y_test,
                self.y_test_pred,
                self.results_dir
            )


        # Output train and test results
        if self.save_raw_data:
            # 检查并确保是一维数据
            y_test_pred_1d = self.y_test_pred
            y_train_pred_1d = self.y_train_pred
            
            # Flatten predictions if they are 2D with second dimension of 1
            if len(y_test_pred_1d.shape) > 1 and y_test_pred_1d.shape[1] == 1:
                y_test_pred_1d = y_test_pred_1d.flatten()
            if len(y_train_pred_1d.shape) > 1 and y_train_pred_1d.shape[1] == 1:
                y_train_pred_1d = y_train_pred_1d.flatten()
            
            test_results = pd.DataFrame(data={"y_test": self.y_test,
                                              "y_test_pred": y_test_pred_1d})
            train_results = pd.DataFrame(data={"y_train": self.y_train,
                                               "y_train_pred": y_train_pred_1d})
            test_results.to_csv(self.results_dir.joinpath("test_results.csv"), index = False)
            train_results.to_csv(self.results_dir.joinpath("train_results.csv"), index = False)    

        return None
