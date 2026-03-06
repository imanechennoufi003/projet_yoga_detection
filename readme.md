<<<<<<< HEAD
# Yoga AI - Web Dashboard

Le projet fonctionne maintenant en interface web (et non uniquement dans le terminal), tout en gardant le concept original:
- detection de poses de yoga avec HOG + SVM,
- comparaison de performance avec fond vs sans fond,
- visualisations IA (accuracy/loss, confusion matrix, precision/recall/F1, distribution des classes).

## Dataset
Le dataset est disponible ici: [Yoga Poses Dataset](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/data)

Structure attendue:
- `data/raw/<pose>/*.jpg|png|jpeg|bmp`
- `data/raw_sans_fond/<pose>/*.png|jpg|...`

Poses par defaut:
- `warrior`
- `downdog`
- `goddess`
- `plank`
- `tree`

## Installation
```bash
pip install -r requirements.txt
```

## Lancer l'application web
Depuis la racine du projet:
```bash
streamlit run src/web_app.py
```

Puis ouvrir l'URL locale fournie par Streamlit (souvent `http://localhost:8501`).

## Fonctions disponibles dans l'UI
- Dashboard principal avec cartes KPI et comparaison des experiences.
- Pose Detection:
  - upload d'image,
  - prediction de la pose,
  - niveau de confiance par classe.
- Model Insights:
  - courbes `accuracy vs loss` (train/test),
  - matrice de confusion,
  - precision/recall/F1 par classe,
  - distribution des classes.
=======
The dataset is available at: [Yogposes][def]

[def]: https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/data
>>>>>>> 427937ed907c2d960273ba1ed53c48ec736d4d2f
