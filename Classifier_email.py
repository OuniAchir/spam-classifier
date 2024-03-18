import urllib.request
import sys
import tarfile
import os
import numpy
import sklearn
import email
import email.policy
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import accuracy_score
from html import unescape
from sklearn.model_selection import train_test_split
import collections
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
import string
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


def download_and_extract_dataset(file_names, urls, download_directory, dataset_type):
    for (file_name, url) in zip(file_names, urls):
        file_path = os.path.join(download_directory, file_name)
        if not os.path.isfile(file_path):
            urllib.request.urlretrieve(url, file_path)
        tar_file = tarfile.open(file_path)
        
        # Remove the path by resetting it
        members = []
        for member in tar_file.getmembers():
            if member.isreg():
                member.name = os.path.basename(member.name) 
                members.append(member)
        tar_file.extractall(path=os.path.join(download_directory, dataset_type), members=members)
        tar_file.close()

root = "https://spamassassin.apache.org/old/publiccorpus/"

ham1_url = root + "20021010_easy_ham.tar.bz2"

ham3_url = root + "20030228_easy_ham_2.tar.bz2"

ham5_url = root + "20030228_hard_ham.tar.bz2"

ham_url = [ham1_url, ham3_url, ham5_url]

ham_filename = ["ham1.tar.bz2", "ham3.tar.bz2", "ham5.tar.bz2"]

spam1_url = root + "20021010_spam.tar.bz2"

spam4_url = root + "20050311_spam_2.tar.bz2"

spam_url = [spam1_url, spam4_url]

spam_filename = ["spam1.tar.bz2", "spam4.tar.bz2"]

path = "./data/"

if not os.path.isdir(path):
 os.makedirs(path)

download_and_extract_dataset(spam_filename, spam_url, path, "spam")

download_and_extract_dataset(ham_filename, ham_url, path, "ham")

def load_emails(directory, filename):
    
    with open(os.path.join(directory, filename), "rb") as f:
      
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
    

ham_filenames = [name for name in sorted(os.listdir("./data/ham")) if name != 'cmds']
spam_filenames = [name for name in sorted(os.listdir("./data/spam")) if name != 'cmds']

ham_emails = [load_emails("./data/ham", filename=name) for name in ham_filenames]
spam_emails = [load_emails("./data/spam", filename=name) for name in spam_filenames]

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))


ham= np.array(ham_emails,dtype=object)
spam= np.array(spam_emails,dtype=object)

#Cette fonction prétraite le corps d'un email
def preprocessing(email_contents):

  # Convert all letters to lowercase
  email_contents=email_contents.lower()
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
    #Remove punctuation
  email_contents = email_contents.translate(str.maketrans('', '', string.punctuation)) 
    # Replace newlines and tabs with a space
  email_contents= re.sub(r'\n|\t', ' ', email_contents)
    
    # Normalize whitespace
  email_contents = re.sub(r'\s+', ' ', email_contents).strip()
  return email_contents

#Convert html to text
def html_to_text(html):

    email_content= re.sub(r'<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    email_content = re.sub(r'<a\s.*?>', ' HYPERLINK ', email_content, flags=re.M | re.S | re.I)
    email_content = re.sub(r'<.*?>', '', email_content, flags=re.M | re.S)
    email_content = re.sub(r'(\s*\n)+', '\n', email_content, flags=re.M | re.S)
    
    return unescape(email_content) 

#Convert email to texte (lisibe)
def email_to_text(email):
    
    html = None
    for entity in email.walk():

        #Some emails have multiple parts, each part is handled separately
        entity_type = entity.get_content_type()
        if not entity_type in ("text/plain", "text/html"):
            continue
        
        try:
            content = entity.get_content()
            #Sometimes this is impossible for encoding reasons
        except: 
            content = str(entity.get_payload())
        if entity_type == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_text(html)

##la fonction qui fait le preprocessig de tous les emails
def preprocess(X): 
  emails_process =[]

  for email_content in X:
    email_content=email_to_text(email_content) or " "
    email_content=preprocessing(email_content)
    emails_process.append(email_content)

  return emails_process

_X=preprocess(X)

#Creation list_vocabulaire
def list_vocabulaire2(X,k):
  list=X.split()
  vocabulaire=[]
  v = collections.Counter(list)
  keys = v.keys() # récupérer les clés de v
  for key in keys:
   if v[key]>k :
    vocabulaire.append(key) #ajouter les mots qui se repetent plus q k fois dans la lste vocabulaire
  return vocabulaire

spam=_X[len(ham_emails):] #retourner les emails spam

vocab = spam
vocab = ''.join(vocab)
v = list_vocabulaire2(vocab,5) # pour le k nous avonss testé ses valeurs et le k=5 donne le meilleur résultat

# Création d'un objet CountVectorizer avec le vocabulaire spécifié
count_vect = CountVectorizer(binary=True, vocabulary=v)

# Transformation des e-mails en une matrice binaire d'occurrences de mots
x_bin = count_vect.transform(_X).toarray()

# Affichage de la matrice binaire d'occurrences de mots
print(x_bin)


x_bin_train, x_bin_test, y_bin_train,y_bin_test  = train_test_split(x_bin, y, random_state=42, test_size = 0.3)

# Création d'un objet CountVectorizer avec le vocabulaire spécifié
count_vect = CountVectorizer(binary=False, vocabulary=v)

# Transformation des e-mails en une matrice binaire d'occurrences de mots
x_comp = count_vect.transform(_X).toarray()

# Affichage de la matrice binaire d'occurrences de mots
print(x_comp)

x_comp_train, x_comp_test, y_comp_train,y_comp_test  = train_test_split(x_comp, y, random_state=42, test_size = 0.3)

def plot_roc_curve(model, X_test, y_test):
    # Obtenir les probabilités de prédiction du modèle sur les données de test
    y_pred_proba = model.predict_proba(X_test)[:, 1]

#Calculer les taux de faux positifs (FPR), les taux de vrais positifs (TPR) et les seuils
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

#Calculer l'aire sous la courbe ROC (AUC)
    roc_auc = auc(fpr, tpr)

#Tracer la courbe ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs (FPR)')
    plt.ylabel('Taux de vrais positifs (TPR)')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    # Obtenir les prédictions du modèle sur les données de test
    y_pred = model.predict(X_test)

#Calculer la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    # Créer une représentation visuelle de la matrice de confusion à l'aide de Seaborn
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Matrice de confusion")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.show()

##une autre fonction pour comparer les modeles en 2 approches vecteurs binaire 
## et vecteurs par comptage

#compraision de courbe_roc_curve
def compare_roc_curve(model1, model2, X_test1, X_test2, y_test1, y_test2):
    # Obtenir les probabilités de prédiction des deux modèles sur les données de test
    y_pred_proba1 = model1.predict_proba(X_test1)[:, 1]
    y_pred_proba2 = model2.predict_proba(X_test2)[:, 1]

##Calculer les taux de faux positifs (FPR), les taux de vrais positifs (TPR) et les seuils pour les deux modèles
    fpr1, tpr1, thresholds1 = roc_curve(y_test1, y_pred_proba1)
    fpr2, tpr2, thresholds2 = roc_curve(y_test2, y_pred_proba2)

#Calculer l'aire sous la courbe ROC (AUC) pour les deux modèles
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)

#Tracer la courbe ROC pour les deux modèles sur le même graphe
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='Approche 1 (AUC = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='blue', lw=2, label='Approche 2 (AUC = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs (FPR)')
    plt.ylabel('Taux de vrais positifs (TPR)')
    plt.title('Courbe ROC - Comparaison de modèles')
    plt.legend(loc="lower right")
    plt.show()


#fct pour comparer les accuracy
def compare_accuracy(model1, model2, X_test1, X_test2, y_test1, y_test2):
    # Obtenir les prédictions du modèle 1 sur les données de test 1
    y_pred1 = model1.predict(X_test1)
    # Calculer l'exactitude du modèle 1
    accuracy1 = accuracy_score(y_test1, y_pred1)

    # Obtenir les prédictions du modèle 2 sur les données de test 2
    y_pred2 = model2.predict(X_test2)
    # Calculer l'exactitude du modèle 2
    accuracy2 = accuracy_score(y_test2, y_pred2)

    # Afficher les résultats
    print("Approche 1 - Exactitude : {:.4f}".format(accuracy1))
    print("Approche 2 - Exactitude : {:.4f}".format(accuracy2))

#nous avons testé pour le C et c=0.1 donne le meilleur resultat

bin_svm_model1 = SVC(C=0.1, kernel="linear")

bin_svm_model1.fit(x_bin_train, y_bin_train)

#predictions:

test_p_bin= bin_svm_model1.predict(x_bin_test)

test_accuracy = (test_p_bin == y_bin_test).mean() * 100

print("test accuracy: {0:.10f}%".format(test_accuracy))

comp_svm_model = SVC(C=0.2, kernel="linear")
comp_svm_model.fit(x_comp_train, y_comp_train)

train_p = comp_svm_model.predict(x_bin_train)
test_p = comp_svm_model.predict(x_bin_test)

test_accuracy2 = (test_p == y_bin_test).mean() * 100

print("test accuracy: {0:.10f}%".format(test_accuracy2))

print("Accuracy :")
compare_accuracy(bin_svm_model1, comp_svm_model, x_bin_test,x_comp_test, y_bin_test, y_comp_test)

# Create a Logistic Regression model
model1 = LogisticRegression(max_iter=1000,C=20)

# Train the model on the training data
model1.fit(x_bin_train, y_bin_train)

# Make predictions on the test data
y_pred = model1.predict(x_bin_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy))
test_accuracy = (y_pred == y_bin_test).mean() * 100
plot_roc_curve(model1,x_bin_test,y_bin_test)
plot_confusion_matrix(model1,x_bin_test, y_bin_test)

# Create a Logistic Regression model
model2 = LogisticRegression(max_iter=1000,C=20)

# Train the model on the training data
model2.fit(x_comp_train, y_comp_train)

# Make predictions on the test data
y_pred2 = model2.predict(x_comp_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy))
test_accuracy = (y_pred2 == y_comp_test).mean() * 100
plot_roc_curve(model2,x_comp_test,y_comp_test)
plot_confusion_matrix(model2,x_comp_test, y_comp_test)

print("Accuracy :")
compare_accuracy(model1, model2, x_bin_test,x_comp_test, y_bin_test, y_comp_test)
print("ROC :")
compare_roc_curve(model1, model2, x_bin_test,x_comp_test, y_bin_test, y_comp_test)

# Create a Decision Tree Classifier
model_dt_bin = DecisionTreeClassifier(max_features=0.8,random_state=42) #l'entrainement du modele nous avons fixé les parametres apres plusieures itérations

# Train the model on the training data
model_dt_bin.fit(x_bin_train, y_bin_train)

# Make predictions on the test data
y_pred = model_dt_bin.predict(x_bin_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy))
test_accuracy = (y_pred == y_bin_test).mean() * 100
plot_roc_curve(model_dt_bin,x_bin_test,y_bin_test)
plot_confusion_matrix(model_dt_bin,x_bin_test, y_bin_test)

# Create a Decision Tree Classifier
model_dt_comp = DecisionTreeClassifier(max_features=0.8,random_state=42)#l'entrainement du modele nous avons fixé les parametres apres plusieures itérations

# Train the model on the training data
model_dt_comp.fit(x_comp_train, y_comp_train)

# Make predictions on the test data
y_pred = model_dt_comp.predict(x_comp_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy))
test_accuracy = (y_pred == y_comp_test).mean() * 100
plot_roc_curve(model_dt_comp,x_comp_test,y_comp_test)
plot_confusion_matrix(model_dt_comp,x_comp_test, y_comp_test)

print("Accuracy :")
compare_accuracy(model_dt_bin, model_dt_comp, x_bin_test,x_comp_test, y_bin_test, y_comp_test)
print("ROC :")
compare_roc_curve(model_dt_bin, model_dt_comp, x_bin_test,x_comp_test, y_bin_test, y_comp_test)

# Create a Decision Tree Classifier
model_rf_bin = RandomForestClassifier(n_estimators=100,random_state=40)

# Train the model on the training data
model_rf_bin.fit(x_bin_train, y_bin_train)

# Make predictions on the test data
y_pred = model_rf_bin.predict(x_bin_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy))
test_accuracy = (y_pred == y_bin_test).mean() * 100
plot_roc_curve(model_rf_bin,x_bin_test,y_bin_test)
plot_confusion_matrix(model_rf_bin,x_bin_test, y_bin_test)


# Create a Decision Tree Classifier
model_rf_comp= RandomForestClassifier(n_estimators=100,random_state=40)

# Train the model on the training data
model_rf_comp.fit(x_comp_train, y_comp_train)

# Make predictions on the test data
y_pred = model_rf_comp.predict(x_comp_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy))
test_accuracy = (y_pred == y_comp_test).mean() * 100
plot_roc_curve(model_rf_comp,x_comp_test,y_comp_test)
plot_confusion_matrix(model_rf_comp,x_comp_test, y_comp_test)

print("Accuracy :")
compare_accuracy(model_rf_bin, model_rf_comp, x_bin_test,x_comp_test, y_bin_test, y_comp_test)
print("ROC :")
compare_roc_curve(model_rf_bin, model_rf_comp, x_bin_test,x_comp_test, y_bin_test, y_comp_test)


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
print("test accuracy: {0:.10f}%".format(test_accuracy))
test_accuracy = (y_pred == y_bin_test).mean() * 100
# Create confusion matrix

confusion_mat = confusion_matrix(y_bin_test, y_pred)

# Plot confusion matrix
labels = ['Non-Spam/Harmless', 'Spam/Harmful']
sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Compute false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_bin_test, y_pred)

# Compute Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

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
print(f"Loss function = {loss:.4f}")
print(f"Accuracy = {accuracy:.4f}")

# Predict on test data
y_pred = model_nn_comp.predict(x_comp_test)
y_pred = np.round(y_pred).flatten()
# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy))
test_accuracy = (y_pred == y_comp_test).mean() * 100
# Create confusion matrix

confusion_mat = confusion_matrix(y_comp_test, y_pred)

# Plot confusion matrix
labels = ['Non-Spam/Harmless', 'Spam/Harmful']
sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Compute false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_comp_test, y_pred)

# Compute Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

print("Accuracy :")
compare_accuracy(model1, model2, x_bin_test,x_comp_test, y_bin_test, y_comp_test)
print("ROC :")
compare_roc_curve(model1, model2, x_bin_test,x_comp_test, y_bin_test, y_comp_test)

def compare_models_app(x,Model1,Model2,Model3,Model4,Model5,x_test,y_test):
 # Créer une liste de noms de modèles
 if x==1 : print("\n approche binaire \n")
 else :print("\n approche par comptage \n")
 noms_modeles = ['SVM', 'Logistic Regression', 'Decision Tree','Random Forest','Neural Network']

# Créer une liste de résultats d'exactitude pour chaque modèle
 resultats_accuracy = [Model1.score(x_test,y_test), Model2.score(x_test,y_test), Model3.score(x_test,y_test),
                       Model4.score(x_test,y_test),Model5.evaluate(x_test,y_test)[1]]

# Créer un DataFrame à partir des listes
 df = pd.DataFrame({'Modèle': noms_modeles, 'Accuracy': resultats_accuracy})

# Afficher le tableau
 print(df)


# Trouver l'index de la ligne avec la meilleure accuracy
 index_meilleur_accuracy = df['Accuracy'].idxmax()

# Récupérer le nom du modèle avec la meilleure accuracy
 modele_meilleur_accuracy = df.loc[index_meilleur_accuracy, 'Modèle']

# Récupérer la meilleure accuracy
 meilleur_accuracy = df.loc[index_meilleur_accuracy, 'Accuracy']

# Afficher le résultat
 print(f"Le modèle avec la meilleure accuracy est {modele_meilleur_accuracy} avec une accuracy de {meilleur_accuracy}")


compare_models_app(1,bin_svm_model1,model1,model_dt_bin,model_rf_bin,model_nn_bin,x_bin_test,y_bin_test)
compare_models_app(2,comp_svm_model,model2,model_dt_comp,model_rf_comp,model_nn_comp,x_comp_test,y_comp_test)


# Evaluation des modèles sur d'autres données 
data1 = pd.read_csv('C:\\Users\\ordi\\Desktop\\spam_ham_dataset.csv')

# Sélectionne toutes les colonnes sauf la dernière
X1 = data1.iloc[:, :-1].values  

# Sélectionne uniquement la dernière colonne
labels = data1.iloc[:, -1].values  

# Convertir en tableaux NumPy
X1_np = np.array(X1)
labels_np = np.array(labels)

print("X =", X1_np[:2, :])
print("Labels =", labels_np[:2])

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

print("\nAvant le preprocessing :\n", X1[0][2])
print("\nAprès le preprocessing :\n", _X1[0])

# Nombre d'étiquettes égales à 0
count_label_0 = np.count_nonzero(labels_np == 0)

# Nombre d'étiquettes égales à 1
count_label_1 = np.count_nonzero(labels_np == 1)

print("Nombre d'étiquettes égales à 0 :", count_label_0)
print("Nombre d'étiquettes égales à 1 :", count_label_1)

spam1 = X1_np[labels_np == 1]

# Initialiser une liste vide pour stocker les résultats
v1 = []

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

print("test accuracy: {0:.10f}%".format(test_accuracy1))

comp_svm_model1 = SVC(C=0.2, kernel="linear")
comp_svm_model1.fit(x_comp1_train, y_comp1_train)

train_p1 = comp_svm_model1.predict(x_bin1_train)
test_p1 = comp_svm_model1.predict(x_bin1_test)

test_accuracy21 = (test_p1 == y_bin1_test).mean() * 100

print("test accuracy: {0:.10f}%".format(test_accuracy21))

print("Accuracy :")
compare_accuracy(bin_svm_model11, comp_svm_model1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)

# Create a Logistic Regression model
model11 = LogisticRegression(max_iter=1000,C=20)

# Train the model on the training data
model11.fit(x_bin1_train, y_bin1_train)

# Make predictions on the test data
y_pred1 = model11.predict(x_bin1_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy1))
test_accuracy1 = (y_pred1 == y_bin1_test).mean() * 100
plot_roc_curve(model11,x_bin1_test,y_bin1_test)
plot_confusion_matrix(model11,x_bin1_test, y_bin1_test)

# Create a Logistic Regression model
model21 = LogisticRegression(max_iter=1000,C=20)

# Train the model on the training data
model21.fit(x_comp1_train, y_comp1_train)

# Make predictions on the test data
y_pred21 = model21.predict(x_comp1_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy1))
test_accuracy1 = (y_pred21 == y_comp1_test).mean() * 100
plot_roc_curve(model21,x_comp1_test,y_comp1_test)
plot_confusion_matrix(model21,x_comp1_test, y_comp1_test)

print("Accuracy :")
compare_accuracy(model11, model21, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)
print("ROC :")
compare_roc_curve(model11, model21, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)

# Create a Decision Tree Classifier
model_dt_bin1 = DecisionTreeClassifier(max_features=0.8,random_state=42) #l'entrainement du modele nous avons fixé les parametres apres plusieures itérations

# Train the model on the training data
model_dt_bin1.fit(x_bin1_train, y_bin1_train)

# Make predictions on the test data
y_pred1 = model_dt_bin1.predict(x_bin1_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy1))
test_accuracy1 = (y_pred1 == y_bin1_test).mean() * 100
plot_roc_curve(model_dt_bin1,x_bin1_test,y_bin1_test)
plot_confusion_matrix(model_dt_bin1,x_bin1_test, y_bin1_test)

# Create a Decision Tree Classifier
model_dt_comp1 = DecisionTreeClassifier(max_features=0.8,random_state=42)#l'entrainement du modele nous avons fixé les parametres apres plusieures itérations

# Train the model on the training data
model_dt_comp1.fit(x_comp1_train, y_comp1_train)

# Make predictions on the test data
y_pred1 = model_dt_comp1.predict(x_comp1_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy1))
test_accuracy1 = (y_pred1 == y_comp1_test).mean() * 100
plot_roc_curve(model_dt_comp1,x_comp1_test,y_comp1_test)
plot_confusion_matrix(model_dt_comp1,x_comp1_test, y_comp1_test)

print("Accuracy :")
compare_accuracy(model_dt_bin1, model_dt_comp1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)
print("ROC :")
compare_roc_curve(model_dt_bin1, model_dt_comp1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)

# Create a Decision Tree Classifier
model_rf_bin1 = RandomForestClassifier(n_estimators=100,random_state=40)

# Train the model on the training data
model_rf_bin1.fit(x_bin1_train, y_bin1_train)

# Make predictions on the test data
y_pred1 = model_rf_bin1.predict(x_bin1_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy1))
test_accuracy1 = (y_pred1 == y_bin1_test).mean() * 100
plot_roc_curve(model_rf_bin1,x_bin1_test,y_bin1_test)
plot_confusion_matrix(model_rf_bin1,x_bin1_test, y_bin1_test)

# Create a Decision Tree Classifier
model_rf_comp1 = RandomForestClassifier(n_estimators=100,random_state=40)

# Train the model on the training data
model_rf_comp1.fit(x_comp1_train, y_comp1_train)

# Make predictions on the test data
y_pred1 = model_rf_comp1.predict(x_comp1_test)

# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy1))
test_accuracy1 = (y_pred1 == y_comp1_test).mean() * 100
plot_roc_curve(model_rf_comp1,x_comp1_test,y_comp1_test)
plot_confusion_matrix(model_rf_comp1,x_comp1_test, y_comp1_test)

print("Accuracy :")
compare_accuracy(model_rf_bin1, model_rf_comp1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)
print("ROC :")
compare_roc_curve(model_rf_bin1, model_rf_comp1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)
    
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
loss1, accuracy1 = model_nn_bin1.evaluate(x_bin1_test, y_bin1_test)
print(f"Loss function = {loss1:.4f}")
print(f"Accuracy = {accuracy1:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
# Predict on test data
y_pred1 = model_nn_bin1.predict(x_bin1_test)
y_pred1 = np.round(y_pred1).flatten()
# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy1))
test_accuracy1 = (y_pred1 == y_bin1_test).mean() * 100
# Create confusion matrix

confusion_mat1 = confusion_matrix(y_bin1_test, y_pred1)

# Plot confusion matrix
labels1 = ['Non-Spam/Harmless', 'Spam/Harmful']
sns.heatmap(confusion_mat1, annot=True, fmt="d", xticklabels=labels1, yticklabels=labels1)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Compute false positive rate, true positive rate, and thresholds
fpr1, tpr1, thresholds1 = roc_curve(y_bin1_test, y_pred1)

# Compute Area Under the Curve (AUC)
roc_auc1 = auc(fpr1, tpr1)

# Plot ROC curve
plt.plot(fpr1, tpr1, label='ROC curve (area = {:.2f})'.format(roc_auc1))
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

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
loss1, accuracy1 = model_nn_comp1.evaluate(x_comp1_test, y_comp1_test)
print(f"Loss function = {loss:.4f}")
print(f"Accuracy = {accuracy:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
# Predict on test data
y_pred1 = model_nn_comp1.predict(x_comp1_test)
y_pred1 = np.round(y_pred1).flatten()
# Evaluate the model's performance
print("test accuracy: {0:.10f}%".format(test_accuracy1))
test_accuracy1 = (y_pred1 == y_comp1_test).mean() * 100
# Create confusion matrix

confusion_mat1 = confusion_matrix(y_comp1_test, y_pred1)

# Plot confusion matrix
labels1 = ['Non-Spam/Harmless', 'Spam/Harmful']
sns.heatmap(confusion_mat1, annot=True, fmt="d", xticklabels=labels1, yticklabels=labels1)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Compute false positive rate, true positive rate, and thresholds
fpr1, tpr1, thresholds1 = roc_curve(y_comp1_test, y_pred1)

# Compute Area Under the Curve (AUC)
roc_auc1 = auc(fpr1, tpr1)

# Plot ROC curve
plt.plot(fpr1, tpr1, label='ROC curve (area = {:.2f})'.format(roc_auc1))
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

print("Accuracy :")
compare_accuracy(model11, model21, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)
print("ROC :")
compare_roc_curve(model11, model21, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)

import pandas as pd
def compare_models_app(x,Model1,Model2,Model3,Model4,x_test,y_test):
 # Créer une liste de noms de modèles
 if x==1 : print("\n approche binaire \n")
 else :print("\n approche par comptage \n")
 noms_modeles = ['SVM', 'Logistic Regression', 'Decision Tree','Random Forest']

# Créer une liste de résultats d'exactitude pour chaque modèle
 resultats_accuracy = [Model1.score(x_test,y_test), Model2.score(x_test,y_test), Model3.score(x_test,y_test),
                       Model4.score(x_test,y_test)]

# Créer un DataFrame à partir des listes
 df = pd.DataFrame({'Modèle': noms_modeles, 'Accuracy': resultats_accuracy})

# Afficher le tableau
 print(df)


# Trouver l'index de la ligne avec la meilleure accuracy
 index_meilleur_accuracy = df['Accuracy'].idxmax()

# Récupérer le nom du modèle avec la meilleure accuracy
 modele_meilleur_accuracy = df.loc[index_meilleur_accuracy, 'Modèle']

# Récupérer la meilleure accuracy
 meilleur_accuracy = df.loc[index_meilleur_accuracy, 'Accuracy']

# Afficher le résultat
 print(f"Le modèle avec la meilleure accuracy est {modele_meilleur_accuracy} avec une accuracy de {meilleur_accuracy}")

 compare_models_app(1,bin_svm_model11,model11,model_dt_bin1,model_rf_bin1,x_bin1_test,y_bin1_test)
compare_models_app(2,comp_svm_model1,model21,model_dt_comp1,model_rf_comp1,x_comp1_test,y_comp1_test)
