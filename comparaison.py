from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

##une autre fonction pour comparer les modeles en 2 approches vecteurs binaire 
## et vecteurs par comptage
from sklearn.metrics import accuracy_score
from html import unescape

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

import pandas as pd

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