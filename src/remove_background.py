import os
from rembg import remove
from PIL import Image
import cv2

input_dir = "../data/raw"
output_dir = "../data/raw_sans_fond"

poses = ['warrior', 'downdog', 'goddess', 'plank', 'tree']

for pose in poses:
    src_folder = os.path.join(input_dir, pose)
    dst_folder = os.path.join(output_dir, pose)
    os.makedirs(dst_folder, exist_ok=True)
    
    print(f"Traitement de {pose}...")
    
    for img_name in os.listdir(src_folder):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(src_folder, img_name)
            output_path = os.path.join(dst_folder, img_name)
            
            try:
                input_image = Image.open(input_path)
                
                # 1. Enlever le fond (retourne une image RGBA avec transparence)
                output_image = remove(input_image)
                
                # 2. CORRECTION : Convertir RGBA -> RGB avec fond blanc pour JPEG
                if output_image.mode == 'RGBA':
                    # Créer un fond blanc
                    background = Image.new('RGB', output_image.size, (255, 255, 255))
                    # Utiliser le canal alpha comme masque pour coller l'image sur le fond blanc
                    background.paste(output_image, mask=output_image.split()[3])
                    output_image = background
                elif output_image.mode in ('LA', 'P'):
                    # Gérer aussi les modes gris avec transparence
                    output_image = output_image.convert('RGB')
                
                # 3. Sauvegarder toujours en PNG pour éviter tout problème de qualité
                # (On change l'extension si nécessaire)
                if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                    output_path = output_path.rsplit('.', 1)[0] + '.png'
                
                output_image.save(output_path)
                print(f"  -> {os.path.basename(output_path)} OK")
                
            except Exception as e:
                print(f"  -> Erreur sur {img_name}: {e}")

print("\n✅ TERMINÉ ! Vérifie le dossier raw_sans_fond")