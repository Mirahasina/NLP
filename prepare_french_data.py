import pandas as pd
import os

# Configuration
PARQUET_URL = "https://huggingface.co/api/datasets/krm/modified-orangeSum/parquet/default/train/0.parquet"
OUTPUT_DIR = "orangesum"
OUTPUT_FILE = "orangesum_test.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

def main():
    print(f"Téléchargement du dataset depuis : {PARQUET_URL}")
    
    # Création du dossier si inexistant
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Dossier {OUTPUT_DIR} créé.")

    try:
        # Lecture du fichier Parquet directement depuis l'URL
        df = pd.read_parquet(PARQUET_URL)
        print(f"Dataset chargé : {len(df)} lignes.")
        
        # Vérification des colonnes (OrangeSum utilise généralement 'text' et 'summary')
        # On les renomme pour correspondre au format du projet (article, highlights)
        mapping = {
            'text': 'article',
            'summary': 'highlights'
        }
        
        # Si les colonnes existent, on les renomme
        found_cols = [c for c in mapping.keys() if c in df.columns]
        if found_cols:
            df = df.rename(columns=mapping)
            print(f"Colonnes renommées : {found_cols}")
        else:
            print(f"Attention : Les colonnes attendues {list(mapping.keys())} n'ont pas été trouvées. Colonnes présentes : {df.columns.tolist()}")

        # On garde uniquement les colonnes nécessaires si elles existent
        keep_cols = ['article', 'highlights']
        df = df[[c for c in keep_cols if c in df.columns]]
        
        # Sauvegarde en CSV
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Dataset sauvegardé avec succès dans : {OUTPUT_PATH}")
        print(f"Aperçu :\n{df.head()}")

    except Exception as e:
        print(f"Erreur lors de la préparation des données : {e}")

if __name__ == "__main__":
    main()
