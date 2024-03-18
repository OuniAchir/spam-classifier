from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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