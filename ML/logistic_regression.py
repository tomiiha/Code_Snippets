
'''
Create df, with 'A' column as category, the rest as random ints.
Model will be terrible - this is a proof of concept on code.
'''
def rand_df():
  import pandas as pd
  import numpy as np
  df = pd.DataFrame(np.random.randint(0,3,size=(500, 1)), columns=list('A'))
  df = df.join(pd.DataFrame(np.random.randint(0,50,size=(500, 3)), columns=list('BCD')))
  return df

def log_reg(df = rand_df()):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    # Instanteate model class.
    logreg = LogisticRegression()

    # Capture columns (X) ...
    feature_cols = ['B','C','D']
    X = df[feature_cols]

    # Used to predict value (y).
    y = df.A

    # Train/Test Split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

    # Fit the Model.
    logreg.fit(X,y)

    # Predict results.
    y_pred_class = logreg.predict(X_test)

    # Show relative accuracy of the model.
    print((metrics.accuracy_score(y_test, y_pred_class)))
