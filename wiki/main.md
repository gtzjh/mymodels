# MyModels


# 编码后的类别变量如何进行SHAP解释

## 对独热编码（one-hot）

可参考下面两条官方的PR

[Questions about SHAP handling categorical variables · Issue #397 · shap/shap](https://github.com/shap/shap/issues/397)

[Methods for combining one hot encoded columns · Issue #1654 · shap/shap](https://github.com/shap/shap/issues/1654)

作者表示可以把编码后的变量累加

另外，有一篇Blog介绍了不同的方式，以及一个新的库名为ACV用于解决独热编码的情况

[salimamoukou/acv00](https://github.com/salimamoukou/acv00)

[The right way to compute your Shapley Values | Towards Data Science](https://towardsdatascience.com/the-right-way-to-compute-your-shapley-values-cfea30509254/)
