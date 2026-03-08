# Yoga AI - Web Dashboard

Le projet detecte des poses de yoga a partir de descripteurs HOG, avec deux classifieurs:
- `SVM`
- `KNN`

Le dashboard et le script terminal comparent aussi les performances avec fond vs sans fond.

## Dataset
Source: [Yoga Poses Dataset](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/data)

Structure attendue:
- `data/raw/<pose>/*.jpg|png|jpeg|bmp`
- `data/raw_sans_fond/<pose>/*.jpg|png|jpeg|bmp`

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

## Lancer le script terminal
Depuis la racine du projet:
```bash
python src/main.py
```

## Fonctions disponibles
- Entrainement HOG + SVM et HOG + KNN.
- Comparaison avec fond vs sans fond.
- Dashboard web avec KPI, prediction d'image et visualisations du modele.
- Export de matrices de confusion et graphiques comparatifs dans `results/plots/`.
