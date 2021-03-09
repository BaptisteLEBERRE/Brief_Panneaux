# Reconnaissance panneau de signialisation

## Remerciements
Avant tout, un grand merci à Paul, Pereg et Thomas qui nous ont permis de passer les 18h d'entrainement du modèle en nous donnant généreusement leur sauvegarde.<br> 
Ils nous ont également aidé par la suite pour la partie de traitement de l'image.

Ensuite, un grand merci à Gilbert Tanner pour son travail sur la détection d'objet dont a été inspiré la majeur partie du travail effectué que ce soit sur 
[youtube](https://www.youtube.com/watch?v=cvyDYdI2nEI) ou encore encore sur son [blog](https://gilberttanner.com/blog/tensorflow-object-detection-with-tensorflow-2-creating-a-custom-model).

## Modèle de classification

Après avoir traité les images, nous avons mis en place un modèle en suivant des exemples d'internet utilisés dans ce genre de projet. Après plus tests, nous avons opté pour
un *learning rate* de 0.001 et 10 epochs (l'apprentissage atteint un plateau vers les 10 epochs).
![courbe_loss](images/courbe_loss.PNG)

Le modèle affiche un val_accuracy de 0.9986 et un score de prédiction de 0.9802 ce qui est plus que satisfaisant.
![epoch](images/epoch.PNG)
