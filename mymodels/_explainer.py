import numpy as np
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
            results_dir: str | pathlib.Path,
            model_object,
            model_name: str,
            used_X_train: pd.DataFrame,
            used_X_test: pd.DataFrame,
            sample_train_k: int | None = None,
            sample_test_k: int | None = None,
            cat_features: list[str] | tuple[str] | None = None,           
        ):
        self.results_dir = pathlib.Path(results_dir)
        self.model_obj = model_object
        self.model_name = model_name
        self.used_X_train = used_X_train
        self.used_X_test = used_X_test
        self.sample_train_k = sample_train_k
        self.sample_test_k = sample_test_k
        self.cat_features = cat_features

    
    def explain(self):
        """Transform categorical data to numerical data
        'Cause the input data for calculating SHAP values must be consistent with the training data,
        so we need to convert the categorical variables in the test data to numerical variables.
        """

        # Sampling or not
        if self.sample_train_k is not None:
            self.used_X_train = shap.sample(self.used_X_train, self.sample_train_k)
        if self.sample_test_k is not None:
            self.used_X_test = shap.sample(self.used_X_test, self.sample_test_k)


        # Set the explainer
        # 这里没有使用shap.Explainer，因为对于xgboost和random forest, 它没有选择TreeExplainer
        if self.model_name in ["svr", "knr", "mlpr", "adar"]:
            _explainer = shap.KernelExplainer(self.model_obj.predict, self.used_X_train)
        elif self.model_name in ["svc", "adac"]:
            # decision_function 本身的含义：
            # 返回样本到决策边界的带符号距离（或决策分数）
            # 在二分类中：
            # 正值：模型倾向于将样本分类为正类（类别 1）
            # 负值：模型倾向于将样本分类为负类（类别 0）
            # 数值大小：模型确信度的度量
            # 同时注意，输出的shap_values的维度是(n_samples, n_features), shap_values的值是指对正样本的贡献
            _explainer = shap.KernelExplainer(self.model_obj.decision_function, self.used_X_train)
        elif self.model_name in ["knc", "mlpc"]:
            # 对于sklearn的knc和mlpc, 因为其模型内部的决策机制是基于概率
            # 所以使用KernelExplainer对它们进行解释的时候，输出的shap_values是概率值
            # 同时注意，输出的shap_values的维度是(n_samples, n_features, n_targets)
            _explainer = shap.KernelExplainer(self.model_obj.predict_proba, self.used_X_train)
        elif self.model_name in ["dtr", "rfr", "gbdtr", "xgbr", "lgbr", "catr",
                                 "dtc", "rfc", "gbdtc", "xgbc", "lgbc", "catc"]:
            # 对sklearn的decision tree和random forest, 因为其模型内部的决策机制是基于概率
            # 所以使用TreeExplainer对它们进行解释的时候，输出的shap_values是概率值
            # 而对于sklearn的gbdt, 以及下面提到的xgboost, lightgbm, catboost, 因为其模型内部的决策机制是基于log-odds空间
            # 所以使用TreeExplainer对它们进行解释的时候，输出的shap_values是log-odds值
            _explainer = shap.TreeExplainer(self.model_obj)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # shap.Explainer object will choose the optimal backend automatically
        self.explanation = _explainer(self.used_X_test)
        self.shap_values = self.explanation.values
        self.shap_base_values = self.explanation.base_values

        # print(self.shap_values)
        # print(self.shap_values.shape)
        # print(self.shap_base_values)

        """
        # 在二分类任务中,
        # TreeExplainer对xgboost, lightgbm, catboost输出的正类的概率值p进行对数几率(log-odds)转换:
        # ln(p/(1 - p))
        # 该值>0代表预测为正类(p>0.5), <0代表预测为负类(p<0.5), =0代表预测为中性(p=0.5),
        # 某一个样本在所有特征的shap_values的和, 再加上base_value(expected_value)即为该样本的预测概率的对数值,
        # 因此shap_values的维度将会是(n_samples, n_features)

        # 二分类模型预测的概率值
        proba = self.model_obj.predict_proba(self.used_X_test)[0]
        print("\n第一个样本的模型预测概率值[阴性, 阳性]:")
        print(proba[0], proba[1])

        # 尝试对两个类别的输出概率进行log-odds转换
        # 对于二分类问题, 正类的对数几率与负类的对数几率符号相反但绝对值相同
        # 即: log-odds(p) = -log-odds(1 - p)
        def log_odds(x):
            return np.log(x / (1 - x))
        print("\n第一个样本的log-odds转换后的概率值[阴性, 阳性]:")
        print(log_odds(proba[0]), log_odds(proba[1]))

        # 由于在xgboost, lightgbm, catboost的内部, 是在log-odds的空间中工作, 并将最后的预测值转换回概率值
        # 在TreeExplainer中, 会回到模型原始的log-odds空间中解释
        # 因此, 输出的shap_values和base_value是log-odds空间中的值
        # 所以, 将每个特征的shap_values值的和, 再加上base_value, 即为预测概率的log-odds值
        feature_sum_i = np.sum(self.shap_values[0, :])
        base_value_i = self.shap_base_values[0]
        sum_i = feature_sum_i + base_value_i
        print("\n第一个样本的shap预测值(shap_values + base_value) | shap_values的和 | base_value")
        print(sum_i, feature_sum_i, base_value_i)

        # log-odds 是 SHAP 值计算的理想空间，能够保证特征贡献的可加性、无界性和一致的解释。
        # 为了展示这一点, 我们可以尝试把shap_values和base_value转换回概率值(sigmoid函数即为log-odds的反函数)
        # 显然, sigmoid_sum_i并不等于模型对正类预测概率值, 并且超出了正常的概率范围([0, 1])
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        sigmoid_feature_sum_i = np.sum([sigmoid(x) for x in self.shap_values[0, :]])
        sigmoid_base_value_i = sigmoid(self.shap_base_values[0])
        sigmoid_sum_i = sigmoid_feature_sum_i + sigmoid_base_value_i
        print("\n第一个样本的shap预测值转为概率值(shap_values + base_value) | shap_values的概率值的和 | base_value的概率值")
        print(sigmoid_sum_i, sigmoid_feature_sum_i, sigmoid_base_value_i)
        """
        
        return None
    

    # 我觉得还应该再加一个总的plot的方法，控制输出的样式（风格）、格式、分辨率等


    def summary_plot(self):
        """Summary Plot
        https://shap.readthedocs.io/en/latest/release_notes.html#release-v0-36-0
        """
        shap.summary_plot(self.shap_values[:, :, 0], self.used_X_test, show = False)
        plt.tight_layout()
        plt.savefig(self.results_dir.joinpath('shap_summary.jpg'), dpi = 500)
        plt.close()
        return None


    def dependence_plot(self):
        """Dependence Plot
        https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html#Using-color-to-highlight-interaction-effects
        """
        _results_dir = self.results_dir.joinpath("dependence_plots")
        _results_dir.mkdir(parents = True, exist_ok = True)
        for i in self.feature_names:
            shap.plots.scatter(self.explanation[:, i], show = False)
            plt.tight_layout()
            plt.savefig(_results_dir.joinpath(i + '.jpg'), dpi = 500)
            plt.close()
        return None
    

    # 这个功能暂时不用
    def partial_dependence_plot(self):
        """Partial Dependence Plot
        Partial Dependence Plot is not supported for categorical features.
        """
        _results_dir = self.results_dir.joinpath("partial_dependence_plots")
        _results_dir.mkdir(parents = True, exist_ok = True)
        for _feature_name in self.feature_names:
            if _feature_name not in self.cat_features:
                shap.partial_dependence_plot(
                    _feature_name,
                    self.model_obj.predict,
                    self.used_X_test,
                    model_expected_value = True,
                    feature_expected_value = False,
                    ice = False,
                    show = False
                )
                plt.tight_layout()
                plt.savefig(_results_dir.joinpath(_feature_name + '.jpg'), dpi = 500)
                plt.close()
        return None 
