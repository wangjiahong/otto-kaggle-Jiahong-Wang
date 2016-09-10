import pandas as pd
import numpy as np
import xgboost as xgb
import os
os.chdir("D:/git_repository/otto-kaggle-Jiahong-Wang")

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
submission = pd.read_csv("input/sampleSubmission.csv")
#target is class_1, ..., class_9 - needs to be converted to 0, ..., 8
train['target'] = train['target'].apply(lambda val: np.int64(val[-1:]))-1

Xy_train = train.as_matrix()
X_train = Xy_train[:,1:-1]
y_train = Xy_train[:,-1:].ravel()

X_test = test.as_matrix()[:,1:]

dtrain = xgb.DMatrix(X_train, y_train, missing=np.NaN)
dtest = xgb.DMatrix(X_test, missing=np.NaN)

params = {"objective": "multi:softprob", "eval_metric": "mlogloss", "booster" : "gbtree",
          "eta": 0.05, "max_depth": 3, "subsample": 0.6,
          "colsample_bytree": 0.7, "num_class": 9, "silent":0}

num_boost_round = 100


gbm = xgb.train(params, dtrain, num_boost_round, verbose_eval= True)
pred = gbm.predict(dtest)

print(gbm.eval(dtrain))
submission.iloc[:,1:] = pred

submission.to_csv("submission.csv", index=False)
print 'submission.csv saved to local file'

%matplotlib inline
 
importance = gbm.get_fscore()
 
fdict = {}
for key, name in enumerate(train.columns[1:-1]):
    fdict['f{0}'.format(key)] = name
 
importance_with_names = []
   
for key, value in importance.items():
    importance_with_names.append((fdict[key], value))
   
   
pd.DataFrame(importance_with_names, columns=['feature', 'fscore']).\
set_index('feature').sort_values(['fscore'], ascending=[0])[:20].\
plot(kind="barh", legend=False, figsize=(6, 10))