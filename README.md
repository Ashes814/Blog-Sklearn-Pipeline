# 利用sklearn实现流程化建模

# 导入数据

```python
# Load our datasets
from sklearn import datasets
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
display(diabetes)
```

## **Diabetes Dataset**

Ten baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements were obtained for each of n = 442 diabetes patients, as well as the response of interest, a quantitative measure of disease progression one year after baseline.

- sex
- bmi body mass index
- bp average blood pressure
- s1 tc, total serum cholesterol
- s2 ldl, low-density lipoproteins
- s3 hdl, high-density lipoproteins
- s4 tch, total cholesterol / HDL
- s5 ltg, possibly log of serum triglycerides level
- s6 glu, blood sugar level

# 观察数据

```python
col_names = ['age', 'sex', 'bmi', 
             'bp', 'tc', 'ldl', 
             'hdl', 'tch', 'ltg', 
             'glu']
diabetes_df = pd.DataFrame(X, columns=col_names)

targed_df = pd.DataFrame(y, columns=['progression'])

display(diabetes_df.head())
display(diabetes_df.info())
display(diabetes_df.describe())

display(targed_df.head())
display(targed_df.info())
display(targed_df.describe())
```

# 建模

```python
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

ela_steps = [('scaler', StandardScaler()),
             ('elasticnet', ElasticNet())]

rf_steps = [('rf', RandomForestRegressor())]

ela_parameters = {'elasticnet__l1_ratio': np.linspace(0, 1, 30)}
rf_parameters = {'rf__n_estimators': [100, 500, 1000]}

ela_cv = GridSearchCV(Pipeline(ela_steps), ela_parameters)
rf_cv = GridSearchCV(Pipeline(rf_steps), rf_parameters)
```