import pickle
import numpy as np
import os

# Numele fișierului creat anterior
FILE_NAME = 'face_db.pkl'

print(f"🔍 Verific fișierul: {FILE_NAME} ...\n")

if not os.path.exists(FILE_NAME):
    print("❌ EROARE: Fișierul nu există! Rulează întâi create_db.py")
    exit()

try:
    with open(FILE_NAME, 'rb') as f:
        database = pickle.load(f)

    print(f"✅ Baza de date încărcată cu succes!")
    print(f"📊 Total persoane înregistrate: {len(database)}")
    print("-" * 50)
    print(f"{'NUME':<20} | {'DIMENSIUNE VECTOR':<15} | {'NORMA (Lungime)'}")
    print("-" * 50)

    for name, vector in database.items():
        # Verificăm dimensiunea (ar trebui să fie 512 sau 128)
        shape = str(vector.shape)

        # Verificăm dacă e normalizat (ar trebui să fie aprox 1.0)
        norm = np.linalg.norm(vector)

        print(f"{name:<20} | {shape:<15} | {norm:.4f}")

    print("-" * 50)

    # Test sumar
    first_key = list(database.keys())[0]
    if database[first_key].shape == (512,) or database[first_key].shape == (128,):
        print("\n✅ Structura pare CORECTĂ pentru inferență!")
    else:
        print("\n⚠️ ATENȚIE: Vectorii au o dimensiune ciudată. Verifică codul de generare.")

except Exception as e:
    print(f"❌ Fișierul este corupt sau nu poate fi citit: {e}")