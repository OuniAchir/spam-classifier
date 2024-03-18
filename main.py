from pre_traitement import preprocess,email_to_text,list_vocabulaire2
from données import X,ham_emails,y
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

# Pré-traitement des données
_X=preprocess(X)
print("\nAvant le preprocessing :\n",email_to_text(X[0]))
print("\nApres le preprocessing :\n",_X[0])

spam=_X[len(ham_emails):] #retourner les emails spam

vocab = spam
vocab = ''.join(vocab)
v = list_vocabulaire2(vocab,5) # pour le k nous avonss testé ses valeurs et le k=5 donne le meilleur résultat

# Création d'un objet CountVectorizer avec le vocabulaire spécifié
count_vect = CountVectorizer(binary=True, vocabulary=v)

# Transformation des e-mails en une matrice binaire d'occurrences de mots
x_bin = count_vect.transform(_X).toarray()

x_bin_train, x_bin_test, y_bin_train,y_bin_test  = train_test_split(x_bin, y, random_state=42, test_size = 0.3)

# Création d'un objet CountVectorizer avec le vocabulaire spécifié
count_vect = CountVectorizer(binary=False, vocabulary=v)

# Transformation des e-mails en une matrice binaire d'occurrences de mots
x_comp = count_vect.transform(_X).toarray()

x_comp_train, x_comp_test, y_comp_train,y_comp_test  = train_test_split(x_comp, y, random_state=42, test_size = 0.3)

from models import test_accuracy_svm,test_accuracy2_svm,bin_svm_model1,comp_svm_model,test_accuracy_LR,model1,model2,test_accuracy_LRC,test_accuracy_DTB,model_dt_comp,model_dt_bin,test_accuracy_DTC,test_accuracy_RFB
from comparaison import compare_accuracy,compare_roc_curve
from visualisaton import plot_confusion_matrix,plot_roc_curve

print("test accuracy SVM approche binaire : {0:.10f}%".format(test_accuracy_svm))

print("test accuracy SVM approche par comptage : {0:.10f}%".format(test_accuracy2_svm))

print("Accuracy SVM :")
compare_accuracy(bin_svm_model1, comp_svm_model, x_bin_test,x_comp_test, y_bin_test, y_comp_test)

# Evaluate the model's performance LR
print("test accuracy Logistic Regression approche binaire: {0:.10f}%".format(test_accuracy_LR))
plot_roc_curve(model1,x_bin_test,y_bin_test)
plot_confusion_matrix(model1,x_bin_test, y_bin_test)

# Evaluate the model's performance
print("test accuracy Logistic Regression approche par compatge: {0:.10f}%".format(test_accuracy_LRC))
plot_roc_curve(model2,x_comp_test,y_comp_test)
plot_confusion_matrix(model2,x_comp_test, y_comp_test)

print("Accuracy Logistic Regression :")
compare_accuracy(model1, model2, x_bin_test,x_comp_test, y_bin_test, y_comp_test)
print("ROC Logistic Regression :")
compare_roc_curve(model1, model2, x_bin_test,x_comp_test, y_bin_test, y_comp_test)

# Evaluate the model's 
print("test accuracy Decision Tree approche binaire: {0:.10f}%".format(test_accuracy_DTB))
plot_roc_curve(model_dt_bin,x_bin_test,y_bin_test)
plot_confusion_matrix(model_dt_bin,x_bin_test, y_bin_test)

print("test  Decision tree approche par comptage : {0:.10f}%".format(test_accuracy_DTC))

plot_roc_curve(model_dt_comp,x_comp_test,y_comp_test)
plot_confusion_matrix(model_dt_comp,x_comp_test, y_comp_test)

print("Accuracy Decision Tree :")
compare_accuracy(model_dt_bin, model_dt_comp, x_bin_test,x_comp_test, y_bin_test, y_comp_test)
print("ROC Decision Tree :")
compare_roc_curve(model_dt_bin, model_dt_comp, x_bin_test,x_comp_test, y_bin_test, y_comp_test)

from models import model_rf_bin,model_rf_comp,test_accuracy_RFC,test_accuracy_RN,confusion_mat

# Evaluate the model's
print("test accuracy Random Forest approche binaire : {0:.10f}%".format(test_accuracy_RFB))
plot_roc_curve(model_rf_bin,x_bin_test,y_bin_test)
plot_confusion_matrix(model_rf_bin,x_bin_test, y_bin_test)

# Evaluate the model's performance
print("test accuracy Random Forest approche par comptage : {0:.10f}%".format(test_accuracy_RFC))
plot_roc_curve(model_rf_comp,x_comp_test,y_comp_test)
plot_confusion_matrix(model_rf_comp,x_comp_test, y_comp_test)

print("Accuracy Random Forest :")
compare_accuracy(model_rf_bin, model_rf_comp, x_bin_test,x_comp_test, y_bin_test, y_comp_test)
print("ROC Random Forest :")
compare_roc_curve(model_rf_bin, model_rf_comp, x_bin_test,x_comp_test, y_bin_test, y_comp_test)

# Evaluate the model's performance
print("test accuracy Réseau des neurones approche binaire : {0:.10f}%".format(test_accuracy_RN))

import seaborn as sns
from models import fpr,tpr,roc_auc

# Plot confusion matrix
labels = ['Non-Spam/Harmless', 'Spam/Harmful']
sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

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

from models import loss,accuracy,test_accuracy_RNC,fpr_c,tpr_c,roc_auc_c 
print(f"Loss function = {loss:.4f}")
print(f"Accuracy = {accuracy:.4f}")

# Evaluate the model's performance
print("test accuracy Rééseau de neurone approche par comptage : {0:.10f}%".format(test_accuracy_RNC))

# Plot confusion matrix
labels = ['Non-Spam/Harmless', 'Spam/Harmful']
sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Plot ROC curve
plt.plot(fpr_c, tpr_c, label='ROC curve (area = {:.2f})'.format(roc_auc_c))
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

print("Accuracy Réseau des neurones approche par comptage :")
compare_accuracy(model1, model2, x_bin_test,x_comp_test, y_bin_test, y_comp_test)
print("ROC Réseau des neurones approche par comptage:")
compare_roc_curve(model1, model2, x_bin_test,x_comp_test, y_bin_test, y_comp_test)

from comparaison import compare_models_app,df
from models import model_nn_bin,model_nn_comp

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

# Deuxieme ensemble de données 
from Evaluation_enemble_donnees_2 import X1_np,labels_np,X1,_X1,test_accuracy1,test_accuracy21,test_accuracy1l,test_accuracy1l2
from Evaluation_enemble_donnees_2 import model11,model21,x_bin1_test,x_comp1_test,y_bin1_test,y_comp1_test
from Evaluation_enemble_donnees_2 import test_accuracy1Db,test_accuracy1Dc,model_dt_bin1,model_dt_comp1,test_accuracy1RB,test_accuracy1RC
from Evaluation_enemble_donnees_2 import model_rf_bin1,model_rf_comp1,loss1,accuracy1RN,test_accuracyrn

print("****************************************** Deuxième ensemble de données ******************************************")
print("X =", X1_np[:2, :])
print("Labels =", labels_np[:2])

print("\nAvant le preprocessing :\n", X1[0][2])
print("\nAprès le preprocessing :\n", _X1[0])

print("test accuracy svm approche binaire  : {0:.10f}%".format(test_accuracy1))
print("test accuracy svm approche par comptage : {0:.10f}%".format(test_accuracy21))

print("test accuracy Logistic Resgression approche binaire : {0:.10f}%".format(test_accuracy1l))
print("test accuracy Logistic Resgression approche par comptage : {0:.10f}%".format(test_accuracy1l2))

print("Accuracy Logistic Regression :")
compare_accuracy(model11, model21, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)
print("ROC LOgistic Regression :")
compare_roc_curve(model11, model21, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)

print("test accuracy Decision Tree approche binaire : {0:.10f}%".format(test_accuracy1Db))
print("test accuracy Decision Tree approche par comptage : {0:.10f}%".format(test_accuracy1Dc))

print("Accuracy  Decision Tree :")
compare_accuracy(model_dt_bin1, model_dt_comp1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)
print("ROC Decision Tree :")
compare_roc_curve(model_dt_bin1, model_dt_comp1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)

print("test accuracy Random Forest approche binaire : {0:.10f}%".format(test_accuracy1RB))
print("test accuracy Random Forest approche par comptage : {0:.10f}%".format(test_accuracy1RC))


print("Accuracy Random Forest :")
compare_accuracy(model_rf_bin1, model_rf_comp1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)
print("ROC Random Forest :")
compare_roc_curve(model_rf_bin1, model_rf_comp1, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)


print(f"Loss function approche binaire = {loss1:.4f}")
print(f"Accuracy  approche binaire= {accuracy1RN:.4f}")

print("test accuracy Réseau des neurones approche binaire : {0:.10f}%".format(test_accuracyrn))

from Evaluation_enemble_donnees_2 import confusion_mat1,fpr1rn,tpr1rn,roc_auc1rn

# Plot confusion matrix
labels1 = ['Non-Spam/Harmless', 'Spam/Harmful']
sns.heatmap(confusion_mat1, annot=True, fmt="d", xticklabels=labels1, yticklabels=labels1)
plt.title("Confusion Matrix approche binaire")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Plot ROC curve
plt.plot(fpr1rn, tpr1rn, label='ROC curve (area = {:.2f})'.format(roc_auc1rn))
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) approche binaire')
plt.legend(loc='lower right')
plt.show()

from Evaluation_enemble_donnees_2 import test_accuracy1rnc,fpr1rnc,tpr1rnc,roc_auc1rnc

# Evaluate the model's performance
print("test accuracy Réseau des neurones approche par comptage : {0:.10f}%".format(test_accuracy1rnc))

# Plot confusion matrix
labels1 = ['Non-Spam/Harmless', 'Spam/Harmful']
sns.heatmap(confusion_mat1, annot=True, fmt="d", xticklabels=labels1, yticklabels=labels1)
plt.title("Confusion Matrix approche par comptage ")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Plot ROC curve
plt.plot(fpr1rnc, tpr1rnc, label='ROC curve (area = {:.2f})'.format(roc_auc1rnc))
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) approche par comptage')
plt.legend(loc='lower right')
plt.show()

print("Accuracy Réseau des neurones :")
compare_accuracy(model11, model21, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)
print("ROC Réseau des neurones :")
compare_roc_curve(model11, model21, x_bin1_test,x_comp1_test, y_bin1_test, y_comp1_test)

from Evaluation_enemble_donnees_2 import compare_models_app_2,bin_svm_model11,comp_svm_model1,df_2

# Afficher le tableau
print(df_2)

# Trouver l'index de la ligne avec la meilleure accuracy
index_meilleur_accuracy_2 = df_2['Accuracy'].idxmax()

# Récupérer le nom du modèle avec la meilleure accuracy
modele_meilleur_accuracy_2 = df_2.loc[index_meilleur_accuracy_2, 'Modèle']

# Récupérer la meilleure accuracy
meilleur_accuracy_2 = df_2.loc[index_meilleur_accuracy_2, 'Accuracy']

# Afficher le résultat
print(f"Le modèle avec la meilleure accuracy est {modele_meilleur_accuracy_2} avec une accuracy de {meilleur_accuracy_2}")

compare_models_app_2(1,bin_svm_model11,model11,model_dt_bin1,model_rf_bin1,x_bin1_test,y_bin1_test)
compare_models_app_2(2,comp_svm_model1,model21,model_dt_comp1,model_rf_comp1,x_comp1_test,y_comp1_test)