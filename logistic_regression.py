from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Instanteate model class.
logreg = LogisticRegression()

# Capture columns (X) ...
feature_cols = ['na','ca']
X = glass[feature_cols]

# Used to predict value (y).
y = glass.household

# Train/Test Split.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

# Fit the Model.
logreg.fit(X,y)

# Predict results.
y_pred_class = logreg.predict(X_test)

# Show relative accuracy of the model.
print((metrics.accuracy_score(y_test, y_pred_class)))
