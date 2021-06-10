import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

def evaluate_dataset(dataset,folds):
    target_name = dataset.columns[-1]
    classes = dataset[target_name].unique()
    y = np.array(dataset[target_name])
    X_dataset = dataset.drop([dataset.target_name], 1)
    X = np.array(pd.get_dummies(X_dataset))

    rf = RandomForestClassifier(random_state=0, n_jobs=6)
    sss = StratifiedKFold(n_splits=folds, shuffle=True)
    CV_results = []
    for train_idx, val_idx in sss.split(X, y):
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, X_val = X[train_idx], X[val_idx]

        model = rf.fit(X_train, y_train)
        pred = model.predict(X_val)

        AUC = roc_auc_score(label_binarize(y_val, classes=classes),
                                  label_binarize(pred, classes=classes))

        CV_results.append(AUC)

    return CV_results.mean()



