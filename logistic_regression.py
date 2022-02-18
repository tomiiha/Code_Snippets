def log_reg(samples = 500, feats = 5, classes = 2):
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import pandas as pd
    import numpy as np
    
    # Create classification dummy dataset.
    X, y = make_classification(n_samples=samples,
                               n_features=feats,
                               n_classes=classes,
                               random_state = 21)

    # Instanteate model class.
    logreg = LogisticRegression()

    # Train/Test Split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21)

    # Fit the Model.
    logreg.fit(X,y)

    # Predict results.
    y_pred_class = logreg.predict(X_test)

    # Show relative accuracy of the model.
    print((metrics.accuracy_score(y_test, y_pred_class)))
