import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,DecisionTreeClassifier,RandomForestClassifier

from main import x_bin_train,y_bin_test,y_bin_train,x_bin_test,x_comp_test,x_comp_train,y_comp_test,y_comp_train
#nous avons testé pour le C et c=0.1 donne le meilleur resultat

bin_svm_model1 = SVC(C=0.1, kernel="linear")

bin_svm_model1.fit(x_bin_train, y_bin_train)

#predictions:

test_p_bin= bin_svm_model1.predict(x_bin_test)
test_accuracy_svm = (test_p_bin == y_bin_test).mean() * 100

comp_svm_model = SVC(C=0.2, kernel="linear")
comp_svm_model.fit(x_comp_train, y_comp_train)

train_p = comp_svm_model.predict(x_bin_train)
test_p = comp_svm_model.predict(x_bin_test)

test_accuracy2_svm = (test_p == y_bin_test).mean() * 100

# Create a Logistic Regression model
model1 = LogisticRegression(max_iter=1000,C=20)

# Train the model on the training data
model1.fit(x_bin_train, y_bin_train)

# Make predictions on the test data
y_pred = model1.predict(x_bin_test)

# Evaluate the model's performance
test_accuracy_LR = (y_pred == y_bin_test).mean() * 100

# Create a Logistic Regression model
model2 = LogisticRegression(max_iter=1000,C=20)

# Train the model on the training data
model2.fit(x_comp_train, y_comp_train)

# Make predictions on the test data
y_pred2 = model2.predict(x_comp_test)

# Evaluate the model's performance
test_accuracy_LRC = (y_pred2 == y_comp_test).mean() * 100

# Create a Decision Tree Classifier
model_dt_bin = DecisionTreeClassifier(max_features=0.8,random_state=42) #l'entrainement du modele nous avons fixé les parametres apres plusieures itérations

# Train the model on the training data
model_dt_bin.fit(x_bin_train, y_bin_train)

# Make predictions on the test data
y_pred = model_dt_bin.predict(x_bin_test)

# Evaluate the model's 
test_accuracy_DTB = (y_pred == y_bin_test).mean() * 100

# Create a Decision Tree Classifier
model_dt_comp = DecisionTreeClassifier(max_features=0.8,random_state=42)#l'entrainement du modele nous avons fixé les parametres apres plusieures itérations

# Train the model on the training data
model_dt_comp.fit(x_comp_train, y_comp_train)

# Make predictions on the test data
y_pred = model_dt_comp.predict(x_comp_test)

# Evaluate the model's performance
test_accuracy_DTC = (y_pred == y_comp_test).mean() * 100


# Create a Decision Tree Classifier
model_rf_bin = RandomForestClassifier(n_estimators=100,random_state=40)

# Train the model on the training data
model_rf_bin.fit(x_bin_train, y_bin_train)

# Make predictions on the test data
y_pred = model_rf_bin.predict(x_bin_test)

# Evaluate the model's performance
test_accuracy_RFB = (y_pred == y_bin_test).mean() * 100

# Create a Decision Tree Classifier
model_rf_comp= RandomForestClassifier(n_estimators=100,random_state=40)

# Train the model on the training data
model_rf_comp.fit(x_comp_train, y_comp_train)

# Make predictions on the test data
y_pred = model_rf_comp.predict(x_comp_test)

# Evaluate the model's performance
test_accuracy_RFC = (y_pred == y_comp_test).mean() * 100

# Build the neural network model
model_nn_bin = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(x_bin_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_nn_bin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_nn_bin.fit(x_bin_train, y_bin_train, epochs=10, batch_size=1)

# Evaluate the model
loss, accuracy = model_nn_bin.evaluate(x_bin_test, y_bin_test)
print(f"Loss function = {loss:.4f}")
print(f"Accuracy = {accuracy:.4f}")

# Predict on test data
y_pred = model_nn_bin.predict(x_bin_test)
y_pred = np.round(y_pred).flatten()
# Evaluate the model's performance
test_accuracy_RN = (y_pred == y_bin_test).mean() * 100

# Create confusion matrix
confusion_mat = confusion_matrix(y_bin_test, y_pred)

from sklearn.metrics import roc_curve, auc
# Compute false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_bin_test, y_pred)

# Compute Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Build the neural network model
model_nn_comp = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(x_comp_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_nn_comp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_nn_comp.fit(x_comp_train, y_comp_train, epochs=10, batch_size=1)

# Evaluate the model
loss, accuracy = model_nn_comp.evaluate(x_comp_test, y_comp_test)
# Predict on test data
y_pred = model_nn_comp.predict(x_comp_test)
y_pred = np.round(y_pred).flatten()

# Create confusion matrix

confusion_mat = confusion_matrix(y_comp_test, y_pred)

# Compute false positive rate, true positive rate, and thresholds
fpr_c, tpr_c, thresholds = roc_curve(y_comp_test, y_pred)

# Compute Area Under the Curve (AUC)
roc_auc_C= auc(fpr, tpr)