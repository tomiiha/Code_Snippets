# Create dummy dataframe for linear regression
def create_df(feats = 3, samples = 500):
    import pandas as pd
    from sklearn.datasets import make_regression
    
    # Random state for reproduction.
    rand_state = 21

    df = make_regression(n_samples=samples, 
                         n_features=feats,
                         random_state=rand_state)
    df = pd.DataFrame(df[0], columns=[str(x) for x in range(0,feats)])
    return df
  
  def linear_regression(df = create_df()):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import numpy as np
    
    # Instanteate LR model.
    lr = LinearRegression()
    
    # Pick y and X for model.
    feats = df.columns[1:].tolist()
    X = df[feats]
    y = df['0']
    
    # Set up train/test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=rand_state)
    
    # Fit the model.
    lr.fit(X_train, y_train)
    
    # Create predicted values, and display RMSE.
    y_pred = lr.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
