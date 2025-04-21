import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mymodels._models import MyModels



def test(model_name):
    model = MyModels(
        model_name = model_name, 
        random_state = 0, 
        cat_features = ['cat_feature_1', 'cat_feature_2']  # Only effective for CatBoost models
    )
    print(model.param_space)
    print(model.static_params)

    return None



if __name__ == "__main__":
    for i in ["mlpc", "rfc", "dtr", "lgbr", "gbdtr", "xgbr", "adac", "svc", "knc", "lr"]:
        test(i)