<div style="text-align: center;">

<h1 align="center">🚀 mymodels : Assemble an efficient interpretable machine learning workflow</h1>

</div>

Feel free to contact me: [gtzjh86@outlook.com](mailto:gtzjh86@outlook.com)


## 🤔 Why `mymodels`?

Interpretable machine learning has gained significant prominence across various fields. Machine learning models are valued for their robust capability to capture complex relationships within data through sophisticated fitting algorithms. Complementing these models, interpretability frameworks provide essential tools for revealing such "black-box" models. These interpretable approaches deliver critical insights by ranking feature importance, identifying nonlinear response thresholds, and analyzing interaction relationships between factors. 

Project `mymodels`, is targeting on building a **tiny, user-friendly, and efficient** workflow, for the scientific researchers and students who are seeking to implement interpretable machine learning in their their research works.

## 👨‍🎓 Prerequisites for Beginners

1. **Python Proficiency**

    - [Python tutorial on W3SCHOOL](https://www.w3schools.com/python/default.asp)
    
    - [Liao Xuefeng's Python Tutorial](https://liaoxuefeng.com/books/python/introduction/index.html)
    
    > **DO REMEMBER**: Make a practical demo project after you finish the above learning to enhance what you have learned (i.e., a tiny web crawler). [Here is one of my practice projects](https://github.com/gtzjh/WundergroundSpider)

2. **Machine Learning Fundamentals**

    - [Stanford CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) provides essential theory.

3. **Technical Skills**

    - Environment management with conda/pip
    - Terminal/Command Line proficiency
    - Version control with Git ([My note about Git](https://github.com/gtzjh/learngit))
  
> The above recommended tutorials are selected based solely on personal experience.

## 🛠️ Environment Setup

**Supported platforms**:

- Windows (X86) - Tested on Windows 10/11
- Linux (X86) - Tested on WSL2.0 (Ubuntu)
- macOS (ARM) - Tested on Apple Silicon (M1)

**Requirements**:
- Python 3.10.X

**Create environment**

```bash
conda env create -f requirement.yml -n mymodels -y
```

**Activate**

```bash
conda activate mymodels
```

## 👉 Try

**Try the Titanic demo first**

- Binary classification: [run_titanic.ipynb](run_titanic.ipynb)

  > Dataset source: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)

**And then try other demos**

- Multi-class classification: [run_obesity.ipynb](run_obesity.ipynb)

  > Dataset source: [Obesity Risk Dataset](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)

- Regression task: [run_housing.ipynb](run_housing.ipynb)

  > Dataset source: [Kaggle Housing Data](https://www.kaggle.com/datasets/jamalshah811/housingdata)


## 🤔 Something You may Interested in

[Supported encode methods](https://github.com/gtzjh/mymodels/wiki#supported-encode-methods)

[Supported models](https://github.com/gtzjh/mymodels/wiki#supported-models)

[Something you should know before implementation](https://github.com/gtzjh/mymodels/wiki#the-users-should-know)