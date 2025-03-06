import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import pathlib

shap.initjs()
matplotlib.use('Agg')
plt.rc('font', family = 'Times New Roman')


class MyExplainer:
    """SHAP Visualization Functions Comparison:
    
    1. shap.partial_dependence_plot:
      - Shows average marginal effect of a feature on model output
      - Similar to traditional Partial Dependence Plot (PDP)
      - X-axis: Feature values, Y-axis: Predicted values
    
    2. shap.dependence_plot:
      - Shows relationship between feature values and SHAP values
      - Automatically detects interactions (color-codes 2nd influential feature)
      - X-axis: Feature values, Y-axis: SHAP values
    
    3. shap.scatter_plot:
      - Basic SHAP value visualization
      - Requires manual specification of x/y axes
      - No automatic interaction detection
    
    Key Differences:
    | Function                | Data Source     | SHAP Values | Auto-Interaction | Output Scale |
    |-------------------------|-----------------|-------------|------------------|--------------|
    | partial_dependence_plot | Raw feature     | ❌          | ❌              | Prediction   |
    | dependence_plot         | Raw + SHAP      | ✅          | ✅              | SHAP         |
    | scatter_plot            | SHAP values     | ✅          | ❌              | SHAP         |
    
    Recommendation: Use dependence_plot for most cases, scatter_plot for custom combinations
    """

    def __init__(
            self,
            model_object,
            model_name: str,
            shap_data: pd.DataFrame,
            results_dir: str | pathlib.Path,
            encoder: object = None,
        ):
        self.model_object = model_object
        self.model_name = model_name
        self.shap_data = shap_data
        self.results_dir = pathlib.Path(results_dir)
        self.encoder = encoder

        # Calculate the SHAP values at the initialization stage.
        self.explainer = self._set_explainer()
        self.shap_values = self._calculate_shap_values()

        """Transform categorical data to numerical data
        因为计算SHAP值时输入的数据要与训练数据保持一致
        所以需要将测试数据中的分类变量转换为数值变量。
        """
        if self.model_name != "cat" and self.encoder is not None:
            self.shap_data = self.encoder.transform(self.shap_data)

    
    def _set_explainer(self):
        if self.model_name in ["svr", "knr", "mlp", "ada"]:
            explainer = shap.KernelExplainer(self.model_object.predict, self.shap_data)
        elif self.model_name in ["cat", "rf", "dt", "lgb", "gbdt", "xgb"]:
            explainer = shap.TreeExplainer(self.model_object)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return explainer
    

    def _calculate_shap_values(self):
        _shap_values = self.explainer.shap_values(self.shap_data)
        return _shap_values


    def summary_plot(self):
        """Summary Plot"""
        shap.summary_plot(self.shap_values, self.shap_data, show = False)
        plt.tight_layout()
        plt.savefig(self.results_dir.joinpath('shap_summary.jpg'), dpi = 500)
        plt.close()
        return None


    def dependence_plot(self):
        """Dependence Plot"""
        _results_dir = self.results_dir.joinpath("dependence_plots")
        _results_dir.mkdir(parents = True, exist_ok = True)
        for _feature_name in self.shap_data.columns:
            shap.dependence_plot(
                _feature_name,
                self.shap_values,
                self.shap_data,
                show = False
            )
            plt.tight_layout()
            plt.savefig(_results_dir.joinpath(_feature_name + '.jpg'), dpi = 500)
            plt.close()
        return None
    

    def partial_dependence_plot(self):
        """Partial Dependence Plot
        Partial Dependence Plot is not supported for categorical features.
        """
        _results_dir = self.results_dir.joinpath("partial_dependence_plots")
        _results_dir.mkdir(parents = True, exist_ok = True)
        for _feature_name in self.shap_data.columns:
            if pd.api.types.is_numeric_dtype(self.shap_data[_feature_name]):
                shap.partial_dependence_plot(
                    _feature_name,
                    self.model_object.predict,
                    self.shap_data,
                    model_expected_value = True,
                    feature_expected_value = False,
                    ice = False,
                    show = False
                )
                plt.tight_layout()
                plt.savefig(_results_dir.joinpath(_feature_name + '.jpg'), dpi = 500)
                plt.close()
        return None 
