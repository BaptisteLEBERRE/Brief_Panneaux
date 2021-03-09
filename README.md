# Reconnaissance panneau de signialisation

## Remerciements
Avant tout, un grand merci à Paul, Pereg et Thomas qui nous ont permis de passer les 18h d'entrainement du modèle en nous donnant généreusement leur sauvegarde.<br> 
Ils nous ont également aidé par la suite pour la partie de traitement de l'image.

Ensuite, un grand merci à Gilbert Tanner pour son travail sur la détection d'objet dont a été inspiré la majeur partie du travail effectué que ce soit sur 
[youtube](https://www.youtube.com/watch?v=cvyDYdI2nEI) ou encore encore sur son [blog](https://gilberttanner.com/blog/tensorflow-object-detection-with-tensorflow-2-creating-a-custom-model).

## Modèle de classification

Après avoir traité les images, nous avons mis en place un [modèle](panneau_classification.ipynb) en suivant des exemples d'internet utilisés dans ce genre de projet. Après plus tests, nous avons opté pour
un *learning rate* de 0.001 et 10 *epochs* (l'apprentissage atteint un plateau vers les 10 epochs).

![courbe_loss](images/courbe_loss.PNG)
<br>
<br>
Le modèle affiche un val_accuracy de 0.9986 et un score de prédiction de 0.9802 ce qui est plus que satisfaisant.

![epoch](images/epoch.PNG)
<br>
<br>
On peut observer qu'il n'y a effectivement aucune erreurs lors du test du modèle

![test_panneau](images/test_panneau.PNG)

## Modèle de détection
Pour cette partie, nous nous sommes appuyé sur le travail de Gilbert Tanner. Plusieurs étapes sont nécessaires.

### 1. Installation

Tout d'abord nous avons cloné la branche principale du repository de [Tensorflow Models](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model)

`git clone https://github.com/tensorflow/models.git`

**Installation du package pour python**

`cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .`



https://github.com/protocolbuffers/protobuf/releases



Pour tester l'installation:

`python object_detection/builders/model_builder_tf2_test.py`

Ce qui devrait nous donner si tout se passe bien:

`...
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 91.767s

OK (skipped=1)
Acquiring data`




### Récupération des images et traitement

Nous avons récupéré le dataset d'images du site de l'[IFN](https://benchmark.ini.rub.de/gtsdb_dataset.html).

Plusieurs traitements on été nécessaires:

* les informations des images (Path, coordonnées des panneaux, classe des panneaux) dans un fichier .txt
* format des images en .ppm

Dans un premier temps nous avons travaillé sur ce fichier .txt. Nous avons:

* créé des colonnes qui renseignent les dimentions des images
* remplacé les classes de panneaux (de 0 à 42) car on veut que ce modèle ne détermine que s'il y a un panneau ou pas <br>

![class_change](images/class_change.PNG)

* remplacé le format de l'image dans le Path pour qu'elles passent de .ppm à .jpeg

![name_change](images/name_change.PNG)

* enregistré la nouvelle database dans un fichier csv

Dans un second temps nous avons procédé au chanement de format pour les images. Pour ce faire nous avons utilisé les modules [os](https://docs.python.org/fr/3/library/os.html), cv2 et [glob](https://docs.python.org/fr/3.6/library/glob.html). 


Nous avons donc récupéré le chemin de chaque image, split le nom de celle-ci et enfin remplacé le .ppm par un .jpeg en rajoutant simplement ce dernier.

![image_change](images/image_change.PNG)

### Création du modèle 


