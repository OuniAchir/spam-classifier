# Projet de Classification des emails Spam 
# Introduction
La gestion des e-mails est cruciale dans notre quotidien, et la distinction entre les spams et les messages légitimes est un défi majeur. Les services de messagerie utilisent désormais l'intelligence artificielle (IA) pour créer des filtres anti-spam plus efficaces. Ce projet explore comment l'IA améliore la classification des spams et simplifie la gestion des e-mails.

# Objectifs
Ce projet vise à appliquer les concepts d'apprentissage artificiel pour développer un classifieur de spams. L'objectif est de résoudre de manière pratique un problème réel en sciences de données en utilisant un ensemble de données spécifique issu du site SpamAssassin et cela en utilisant les deux approches machine learning et deep learning et la mise en pratique des différents mesures de performances (métriques).

# Description des données
Les données sont obtenues à partir du site SpamAssassin ou de Kaggle. Ces données sont essentielles pour l'entraînement du détecteur de spams.

# Préparation des données
Le prétraitement des e-mails est crucial. Une fonction spécifique est mise en place pour effectuer des transformations telles que la conversion en minuscules, la suppression des balises HTML, la normalisation des URL, et d'autres. Le vocabulaire est construit, et les caractéristiques sont extraites en utilisant des représentations binaires et par comptage.

# Classification
Cinq modèles sont implémentés pour la classification :

Support Vector Machine (SVM)
Régression Logistique
Arbres de Décision
Forêt Aléatoire
Réseau de Neurones
Deux représentations différentes sont utilisées : vecteur binaire et vecteur de compteurs. Les modèles sont évalués à l'aide de courbes ROC et de matrices de confusion ainsi l'accuracy.

# Comparaison
Des fonctions de  comparaison des approches (vecteur binaire vs vecteur de compteurs) pour chaque modèle est effectuée. Et aussi une fonction comparaison de tous les modèles est implementée tel que les performances sont évaluées en termes d'exactitude.

# Comparaison des performences 
Nous avons examiné les performances de nous modèles sur d'autres données, après l'application des mêmes étapes détaillées précédemment, nous avons obtenue les résultats de test mais satisfaisantes > 0.90 en terme d'accuracy.

# Résultats et Conclusion
Les résultats montrent que certains modèles, tels que SVM et Régression Logistique, ont des performances élevées. L'approche binaire semble donner de meilleurs résultats globaux. Le projet conclut en évaluant les modèles sur d'autres ensembles de données, soulignant l'importance de comprendre la nature des données pour ajuster les modèles en conséquence.