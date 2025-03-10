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

        # Statement the shap values
        self.shap_values = None

        return None
        
    
    def explain(self):
        """Transform categorical data to numerical data
        Because the input data for calculating SHAP values must be consistent with the training data,
        so we need to convert the categorical variables in the test data to numerical variables.
        """
        if self.model_name != "catr" and self.model_name != "catc":
            if self.encoder is not None:
                self.shap_data = self.encoder.transform(self.shap_data)

        # Set the explainer
        if self.model_name in ["svr", "knr", "mlpr", "adar"]:
            _explainer = shap.KernelExplainer(self.model_object.predict, self.shap_data)
        elif self.model_name in ["catr", "catc", "rfr", "dtr", "gbdtr", "xgbr", "lgbr"]:
            _explainer = shap.TreeExplainer(self.model_object)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Get shap value
        self.shap_values = _explainer(self.shap_data).values

        return None


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
