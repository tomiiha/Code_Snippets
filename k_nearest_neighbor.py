def best_knn(samples = 500, feats = 5, classes = 2):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import pandas as pd
    import numpy as np
    
    # Random state for reproduction.
    rand_state = 15
    
    # Create classification dummy dataset.
    X, y = make_classification(n_samples=samples, 
                             n_features=feats,
                             n_classes=classes,
                             random_state=rand_state)
        
    # Create a for loop to calculate the TRAINING ERROR and TESTING ERROR for K=1 through the square root of df.
    n_sqr = int(np.sqrt(samples))
    k_range = list(range(1, n_sqr + 1))
    training_error = []
    testing_error = []

    # Find test accuracy for all values of K between 1 and square root of n (inclusive).
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rand_state)
        knn.fit(X_train, y_train)

        # Calculate training error (error = 1 - accuracy).
        y_pred_class = knn.predict(X_train)
        training_accuracy = metrics.accuracy_score(y_train, y_pred_class)
        training_error.append(1 - training_accuracy)

        # Calculate testing error.
        y_pred_class = knn.predict(X_test)
        testing_accuracy = metrics.accuracy_score(y_test, y_pred_class)
        testing_error.append(1 - testing_accuracy)

    # Create a DataFrame of K, training error, and testing error.
    column_dict = {'K': k_range, 'training error':training_error, 'testing error':testing_error}
    df = pd.DataFrame(column_dict).set_index('K').sort_index(ascending=False)
    return df.sort_values('testing error').head()
