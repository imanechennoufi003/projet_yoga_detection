import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import sys
import warnings
import pandas as pd  # Pour le tableau comparatif

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_images_from_folder, extract_hog_features, save_hog_visualization

def train_and_evaluate(data_path, poses, experiment_name):
    """Fonction qui entraîne et évalue le modèle sur un dataset donné"""
    print(f"\n{'='*60}")
    print(f"🧪 EXPÉRIENCE : {experiment_name}")
    print(f"📁 Chemin : {data_path}")
    print(f"{'='*60}")
    
    X = []
    y = []
    valid_poses = []
    
    # Chargement des images
    for label, pose_name in enumerate(poses):
        folder_path = os.path.join(data_path, pose_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Dossier {pose_name} introuvable")
            continue
            
        images = load_images_from_folder(folder_path)
        if len(images) == 0:
            continue
            
        features, hog_imgs = extract_hog_features(images)
        X.append(features)
        y.extend([label] * len(features))
        valid_poses.append(pose_name)
        
        if len(hog_imgs) > 0 and experiment_name == "Avec Fond (Original)":
            save_path = os.path.join('..', 'results', 'plots', f'hog_{pose_name}.png')
            save_hog_visualization(hog_imgs[0], save_path)
    
    if len(X) == 0:
        print(f"❌ Aucune image trouvée pour {experiment_name}")
        return None, None, None
    
    X = np.vstack(X)
    y = np.array(y)
    
    print(f"📊 Total images : {len(X)} | Classes : {valid_poses}")
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement SVM
    print("🔄 Entraînement du modèle SVM...")
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    
    # Évaluation
    print("🔍 Évaluation...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✅ PRÉCISION : {accuracy * 100:.2f}%")
    
    # Sauvegarde matrice de confusion (seulement pour la dernière expérience)
    if experiment_name == "Sans Fond (Traitée)":
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Matrice de Confusion - {experiment_name}')
        plt.colorbar()
        tick_marks = np.arange(len(valid_poses))
        plt.xticks(tick_marks, valid_poses, rotation=45)
        plt.yticks(tick_marks, valid_poses)
        plt.tight_layout()
        plt.ylabel('Vrai Label')
        plt.xlabel('Label Prédit')
        cm_path = os.path.join('..', 'results', 'plots', f'confusion_{experiment_name.replace(" ", "_")}.png')
        plt.savefig(cm_path)
        print(f"📈 Matrice sauvegardée : {cm_path}")
    
    return accuracy, valid_poses, model

def main():
    print("🧘 PROJET YOGA AI - Étude de l'impact de l'arrière-plan")
    
    # Configuration des poses (tes vrais dossiers)
    poses = ['warrior', 'downdog', 'goddess', 'plank', 'tree']
    
    # Liste des expériences à lancer
    experiments = [
        {
            "name": "Avec Fond (Original)",
            "path": os.path.join('..', 'data', 'raw')
        },
        {
            "name": "Sans Fond (Traitée)",
            "path": os.path.join('..', 'data', 'raw_sans_fond')  # Dossier créé par rembg
        }
    ]
    
    results = []
    
    # Lancer chaque expérience
    for exp in experiments:
        # Vérifier si le dossier existe avant de lancer
        if os.path.exists(exp["path"]):
            accuracy, valid_poses, model = train_and_evaluate(
                exp["path"], poses, exp["name"]
            )
            if accuracy is not None:
                results.append({
                    "Expérience": exp["name"],
                    "Précision (%)": f"{accuracy * 100:.2f}",
                    "Images": "Voir logs ci-dessus"
                })
        else:
            print(f"\n⚠️ Dossier non trouvé : {exp['path']}")
            print(f"   → Cette expérience sera ignorée")
    
    # 🎯 AFFICHAGE DU TABLEAU COMPARATIF (Pour ta présentation !)
    if len(results) >= 2:
        print(f"\n\n{'🏆'*30}")
        print("📊 RÉSULTATS COMPARATIFS - IMPACT DU FOND")
        print(f"{'🏆'*30}")
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        # Calcul de l'amélioration
        acc_with = float(results[0]["Précision (%)"].replace('%', ''))
        acc_without = float(results[1]["Précision (%)"].replace('%', ''))
        improvement = acc_without - acc_with
        
        print(f"\n📈 Amélioration : {'+' if improvement >= 0 else ''}{improvement:.2f}%")
        
        if improvement > 0:
            print("✅ Conclusion : Enlever le fond AMÉLIORE la reconnaissance HOG")
        elif improvement < 0:
            print("⚠️ Conclusion : Le fond n'a pas d'impact négatif majeur (ou HOG est robuste)")
        else:
            print("➡️ Conclusion : L'impact du fond est neutre sur ce dataset")
            
        # Sauvegarder le tableau en image pour la présentation
        plt.figure(figsize=(10, 4))
        plt.axis('tight')
        plt.axis('off')
        table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        plt.title("Comparaison : Impact de l'arrière-plan sur la reconnaissance Yoga", fontsize=14, pad=20)
        
        save_path = os.path.join('..', 'results', 'plots', 'comparaison_fond.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"💾 Tableau comparatif sauvegardé : {save_path}")
        
    elif len(results) == 1:
        print(f"\n⚠️ Une seule expérience a réussi. Précision : {results[0]['Précision (%)']}")
    else:
        print("\n❌ Aucune expérience n'a pu être lancée. Vérifie tes dossiers.")
    
    print(f"\n{'✅'*30}")
    print("FIN DU TRAITEMENT - Bonne présentation ! 🎓")
    print(f"{'✅'*30}")

if __name__ == "__main__":
    main()