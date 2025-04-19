# The Users Should Know

- This project is intended solely for scientific reference. It may contain calculation errors or logical inaccuracies. Users are responsible for verifying the accuracy of the results independently, and the author shall not be held liable for any consequences arising from the use of this code.

- Due to the developer's limited personal capabilities and time constraints, the project may inevitably have shortcomings. We sincerely welcome fellow professionals to provide critiques and suggestions for improvement.

- The project is not suitable for time-series tasks.

- The project is not supported for GPU acceleratation currently.

- The hyperparameters shown in `models.py` are only for demonstration purposes. Users should try different hyperparameters in their actual applications to ensure the robustness of their results.

- The `random_state` is set to `0` for demonstration purposes only. Users should try different `random_state` in their actual applications to ensure the robustness of their results.

- The explanation in this project is currently based on [SHAP](https://shap.readthedocs.io/en/latest/index.html) and PDP (Partial Dependence Plot), Other explanation methods are under developing. 

- Note that explanations may not always be meaningful for real-world tasks, especially after data engineering. Users are solely responsible for validating the appropriateness of explanation methods for their specific use cases.

- The Partial Dependence Plot (PDP) is not supported for classification tasks currently.

- The Partial Dependence Plot (PDP) is not supported for categorical features currently.