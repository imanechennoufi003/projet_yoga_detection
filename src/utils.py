import cv2
import os
from skimage.feature import hog
import numpy as np

def load_images_from_folder(folder_path):
    """Charge toutes les images d'un dossier et retourne une liste"""
    images = []
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Redimensionnement obligatoire pour HOG (taille fixe)
                img = cv2.resize(img, (128, 64))
                images.append(img)
    return images

def extract_hog_features(images):
    """Calcule les descripteurs HOG pour une liste d'images"""
    features_list = []
    hog_images_list = [] # Pour la visualisation
    
    for img in images:
        # Calcul HOG
        features, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, feature_vector=True)
        features_list.append(features)
        hog_images_list.append(hog_image)
        
    return np.array(features_list), hog_images_list

def save_hog_visualization(hog_image, save_path):
    """Sauvegarde une image HOG pour la présentation"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.imshow(hog_image, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()