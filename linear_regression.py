def linear_regression(samples = 500, feats = 3):
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import numpy as np
    
    # Instanteate LR model.
    lr = LinearRegression()
    
    # Random state for reproduction.
    rand_state = 21
    
    X, y = make_regression(n_samples=samples,
                           n_features=feats,
                           random_state=rand_state)
    
    # Set up train/test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=rand_state)
    
    # Fit the model.
    lr.fit(X_train, y_train)
    
    # Create predicted values, and display RMSE.
    y_pred = lr.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
