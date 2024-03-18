import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
import string
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Evaluation des modèles sur d'autres données 
data1 = pd.read_csv('C:\\Users\\ordi\\Desktop\\spam_ham_dataset.csv')

# Sélectionne toutes les colonnes sauf la dernière
X1 = data1.iloc[:, :-1].values  

# Sélectionne uniquement la dernière colonne
labels = data1.iloc[:, -1].values  

# Convertir en tableaux NumPy
X1_np = np.array(X1)
labels_np = np.array(labels)

import re
import string
from nltk.stem import PorterStemmer

def preprocessing(email_contents):
    # Convert all letters to lowercase
    email_contents = email_contents.lower()
    
    # Remove HTML tags
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    
    # Normalize URLs
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    
    # Normalize email addresses
    email_contents = re.sub('\S+@\S+', 'emailaddr', email_contents)
    
    # Normalize numbers
    email_contents = re.sub('\d+', 'nombre', email_contents)
    
    # Normalize dollar signs
    email_contents = re.sub('\$', 'dollar', email_contents)
    
    # Stem words
    stemmer = PorterStemmer()
    words = re.findall('\w+', email_contents)
    stemmed_words = [stemmer.stem(word) for word in words]
    email_contents = ' '.join(stemmed_words)
    
    # Remove non-words and punctuation, replace white spaces with a single space
    
    # Replace non-word characters with a space
    email_contents = re.sub(r'\W+', ' ', email_contents)
    
    # Remove punctuation
    email_contents = email_contents.translate(str.maketrans('', '', string.punctuation))
    
    # Replace newlines and tabs with a space
    email_contents = re.sub(r'\n|\t', ' ', email_contents)
    
    # Normalize whitespace
    email_contents = re.sub(r'\s+', ' ', email_contents).strip()
    
    return email_contents

# Appliquer la fonction preprocessing à chaque élément de X1
_X1 = [preprocessing(email[2]) for email in X1]

# Nombre d'étiquettes égales à 0
count_label_0 = np.count_nonzero(labels_np == 0)

# Nombre d'étiquettes égales à 1
count_label_1 = np.count_nonzero(labels_np == 1)

spam1 = X1_np[labels_np == 1]

# Initialiser une liste vide pour stocker les résultats
v1 = []

from pre_traitement import list_vocabulaire2

# Parcourir chaque élément du tableau NumPy et appliquer la fonction
for item in spam1:
    # Convertir l'élément en chaîne de caractères si ce n'est pas déjà le cas
    item_str = str(item)
    # Appliquer la fonction et ajouter le résultat à la liste
    v1.extend(list_vocabulaire2(item_str, 5))

# Convertir la liste en un ensemble pour éliminer les doublons, si nécessaire
v1 = set(v1)

# Création d'un objet CountVectorizer avec le vocabulaire spécifié
count_vect1 = CountVectorizer(binary=True, vocabulary=v1)

# Transformation des e-mails en une matrice binaire d'occurrences de mots
x_bin1 = count_vect1.transform(_X1).toarray()

x_bin1.shape

x_bin1_train, x_bin1_test, y_bin1_train,y_bin1_test  = train_test_split(x_bin1, labels, random_state=42, test_size = 0.3)
x_bin1_train.shape

# Création d'un objet CountVectorizer avec le vocabulaire spécifié
count_vect1 = CountVectorizer(binary=False, vocabulary=v1)

# Transformation des e-mails en une matrice binaire d'occurrences de mots
x_comp1 = count_vect1.transform(_X1).toarray()

# Affichage de la matrice binaire d'occurrences de mots
print(x_comp1)

x_comp1_train, x_comp1_test, y_comp1_train,y_comp1_test  = train_test_split(x_comp1, labels_np, random_state=42, test_size = 0.3)

#nous avons testé pour le C et c=0.1 donne le meilleur resultat

bin_svm_model11 = SVC(C=0.1, kernel="linear")

bin_svm_model11.fit(x_bin1_train, y_bin1_train)

#predictions:

test_p_bin1= bin_svm_model11.predict(x_bin1_test)

test_accuracy1 = (test_p_bin1 == y_bin1_test).mean() * 100

comp_svm_model1 = SVC(C=0.2, kernel="linear")
comp_svm_model1.fit(x_comp1_train, y_comp1_train)

train_p1 = comp_svm_model1.predict(x_bin1_train)
test_p1 = comp_svm_model1.predict(x_bin1_test)

test_accuracy21 = (test_p1 == y_bin1_test).mean() * 100

from comparaison import compare_accuracy

compare_accuracy(bin_svm_model11, comp_svm_model1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)

# Create a Logistic Regression model
model11 = LogisticRegression(max_iter=1000,C=20)

# Train the model on the training data
model11.fit(x_bin1_train, y_bin1_train)

# Make predictions on the test data
y_pred1 = model11.predict(x_bin1_test)

from visualisaton import plot_confusion_matrix,plot_roc_curve

# Evaluate the model's performance
test_accuracy1l = (y_pred1 == y_bin1_test).mean() * 100
plot_roc_curve(model11,x_bin1_test,y_bin1_test)
plot_confusion_matrix(model11,x_bin1_test, y_bin1_test)

# Create a Logistic Regression model
model21 = LogisticRegression(max_iter=1000,C=20)

# Train the model on the training data
model21.fit(x_comp1_train, y_comp1_train)

# Make predictions on the test data
y_pred21 = model21.predict(x_comp1_test)

# Evaluate the model's performance
test_accuracy1l2 = (y_pred21 == y_comp1_test).mean() * 100
plot_roc_curve(model21,x_comp1_test,y_comp1_test)
plot_confusion_matrix(model21,x_comp1_test, y_comp1_test)

from comparaison import compare_roc_curve

# Create a Decision Tree Classifier
model_dt_bin1 = DecisionTreeClassifier(max_features=0.8,random_state=42) #l'entrainement du modele nous avons fixé les parametres apres plusieures itérations

# Train the model on the training data
model_dt_bin1.fit(x_bin1_train, y_bin1_train)

# Make predictions on the test data
y_pred1 = model_dt_bin1.predict(x_bin1_test)

# Evaluate the model's performance
test_accuracy1Db = (y_pred1 == y_bin1_test).mean() * 100
plot_roc_curve(model_dt_bin1,x_bin1_test,y_bin1_test)
plot_confusion_matrix(model_dt_bin1,x_bin1_test, y_bin1_test)

# Create a Decision Tree Classifier
model_dt_comp1 = DecisionTreeClassifier(max_features=0.8,random_state=42)#l'entrainement du modele nous avons fixé les parametres apres plusieures itérations

# Train the model on the training data
model_dt_comp1.fit(x_comp1_train, y_comp1_train)

# Make predictions on the test data
y_pred1 = model_dt_comp1.predict(x_comp1_test)

# Evaluate the model's performance
test_accuracy1Dc = (y_pred1 == y_comp1_test).mean() * 100
plot_roc_curve(model_dt_comp1,x_comp1_test,y_comp1_test)
plot_confusion_matrix(model_dt_comp1,x_comp1_test, y_comp1_test)


# Create a Decision Tree Classifier
model_rf_bin1 = RandomForestClassifier(n_estimators=100,random_state=40)

# Train the model on the training data
model_rf_bin1.fit(x_bin1_train, y_bin1_train)

# Make predictions on the test data
y_pred1 = model_rf_bin1.predict(x_bin1_test)

# Evaluate the model's performance
test_accuracy1RB = (y_pred1 == y_bin1_test).mean() * 100
plot_roc_curve(model_rf_bin1,x_bin1_test,y_bin1_test)
plot_confusion_matrix(model_rf_bin1,x_bin1_test, y_bin1_test)

# Create a Decision Tree Classifier
model_rf_comp1 = RandomForestClassifier(n_estimators=100,random_state=40)

# Train the model on the training data
model_rf_comp1.fit(x_comp1_train, y_comp1_train)

# Make predictions on the test data
y_pred1 = model_rf_comp1.predict(x_comp1_test)

# Evaluate the model's performance
test_accuracy1RC = (y_pred1 == y_comp1_test).mean() * 100
plot_roc_curve(model_rf_comp1,x_comp1_test,y_comp1_test)
plot_confusion_matrix(model_rf_comp1,x_comp1_test, y_comp1_test)
    
from models import x_bin_train

# Build the neural network model
model_nn_bin1 = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(x_bin_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_nn_bin1.compile(optimizer='adam', loss1='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_nn_bin1.fit(x_bin1_train, y_bin1_train, epochs=10, batch_size=1)

# Evaluate the model
loss1, accuracy1RN = model_nn_bin1.evaluate(x_bin1_test, y_bin1_test)

import matplotlib.pyplot as plt
import seaborn as sns

# Predict on test data
y_pred1 = model_nn_bin1.predict(x_bin1_test)
y_pred1 = np.round(y_pred1).flatten()
# Evaluate the model's performance
test_accuracyrn = (y_pred1 == y_bin1_test).mean() * 100
# Create confusion matrix

confusion_mat1 = confusion_matrix(y_bin1_test, y_pred1)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Compute false positive rate, true positive rate, and thresholds
fpr1rn, tpr1rn, thresholds1rn = roc_curve(y_bin1_test, y_pred1)

# Compute Area Under the Curve (AUC)
roc_auc1rn = auc(fpr1rn, tpr1rn)

from models import x_comp_train

# Build the neural network model
model_nn_comp1 = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(x_comp_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_nn_comp1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_nn_comp1.fit(x_comp1_train, y_comp1_train, epochs=10, batch_size=1)

# Evaluate the model
loss1rnc, accuracy1rnc = model_nn_comp1.evaluate(x_comp1_test, y_comp1_test)
print(f"Loss function = {loss1rnc:.4f}")
print(f"Accuracy = {accuracy1rnc:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
# Predict on test data
y_pred1 = model_nn_comp1.predict(x_comp1_test)
y_pred1 = np.round(y_pred1).flatten()
# Evaluate the model's performance
test_accuracy1rnc = (y_pred1 == y_comp1_test).mean() * 100
# Create confusion matrix

confusion_mat1 = confusion_matrix(y_comp1_test, y_pred1)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute false positive rate, true positive rate, and thresholds
fpr1rnc, tpr1rnc, thresholds1rnc = roc_curve(y_comp1_test, y_pred1)

# Compute Area Under the Curve (AUC)
roc_auc1rnc = auc(fpr1rnc, tpr1rnc)

import pandas as pd
def compare_models_app_2(x,Model1,Model2,Model3,Model4,x_test,y_test):
 # Créer une liste de noms de modèles
 if x==1 : print("\n approche binaire \n")
 else :print("\n approche par comptage \n")
 noms_modeles = ['SVM', 'Logistic Regression', 'Decision Tree','Random Forest']

# Créer une liste de résultats d'exactitude pour chaque modèle
 resultats_accuracy = [Model1.score(x_test,y_test), Model2.score(x_test,y_test), Model3.score(x_test,y_test),
                       Model4.score(x_test,y_test)]

# Créer un DataFrame à partir des listes
 df_2 = pd.DataFrame({'Modèle': noms_modeles, 'Accuracy': resultats_accuracy})