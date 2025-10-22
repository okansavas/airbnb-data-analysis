# run_full_project.py - GÜNCELLENMİŞ (Streamlit otomatik başlatma)
import os
import subprocess
import time
import sys
import webbrowser
from threading import Thread

# ========== Konfiguration ==========
NOTEBOOKS = [
    "notebooks/01_eda_airbnb.ipynb",
    "notebooks/02_preprocessing_airbnb.ipynb", 
    "notebooks/03_modeling_airbnb.ipynb",
    "notebooks/04_training_airbnb.ipynb",
    "notebooks/05_evaluation_airbnb.ipynb",
]

STREAMLIT_APP = "notebooks/06_streamlit_airbnb.py"

# ========== Funktionen ==========
def run_notebook(nb_path: str):
    """Führt ein Jupyter-Notebook aus"""
    print(f"\n📘 Starte Notebook: {nb_path}")
    start = time.time()
    try:
        result = subprocess.run([
            "jupyter", "nbconvert", "--to", "notebook", "--execute",
            nb_path, "--inplace", "--ExecutePreprocessor.timeout=600"
        ], capture_output=True, text=True, check=True)
        
        end = time.time()
        print(f"✅ Erfolgreich abgeschlossen ({round(end - start, 2)}s): {nb_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler beim Ausführen von {nb_path}: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def start_streamlit():
    """Startet die Streamlit-App im Hintergrund"""
    print(f"\n🌐 Starte Streamlit App: {STREAMLIT_APP}")
    try:
        # Streamlit im Hintergrund starten
        process = subprocess.Popen([
            "streamlit", "run", STREAMLIT_APP, 
            "--server.port", "8501",
            "--server.headless", "true"
        ])
        
        # Kurz warten bis Server startet
        time.sleep(5)
        
        # Browser öffnen
        print("🔗 Öffne Browser...")
        webbrowser.open("http://localhost:8501")
        
        print("✅ Streamlit App läuft auf http://localhost:8501")
        print("ℹ️  Drücken Sie Str+C um die App zu beenden")
        
        return process
    except Exception as e:
        print(f"❌ Fehler beim Starten von Streamlit: {e}")
        return None

# ========== Pipeline starten ==========
print("🚀 Starte vollständige Ausführung des Airbnb-Projekts...")

# 1. Alle Notebooks ausführen
all_success = True
for nb in NOTEBOOKS:
    success = run_notebook(nb)
    if not success:
        print(f"❌ Projekt wurde wegen Fehler in {nb} abgebrochen.")
        all_success = False
        break
    time.sleep(2)

if all_success:
    print("\n🎯 ALLE NOTEBOOKS ERFOLGREICH AUSGEFÜHRT!")
    
    # 2. Modellprüfung
    model_path = "models/airbnb_model.pkl"
    if os.path.exists(model_path):
        print(f"✅ Modell erfolgreich gefunden: {model_path}")
    else:
        print(f"⚠️ Kein Modell gefunden unter: {model_path}")
    
    # 3. Streamlit starten
    print("\n" + "="*50)
    print("💻 STARTER STREAMLIT DASHBOARD")
    print("="*50)
    
    streamlit_process = start_streamlit()
    
    if streamlit_process:
        try:
            # Warten bis der Prozess beendet wird
            streamlit_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Streamlit wird beendet...")
            streamlit_process.terminate()
    else:
        print("❌ Streamlit konnte nicht gestartet werden")
        print("ℹ️  Sie können Streamlit manuell starten mit:")
        print(f"   streamlit run {STREAMLIT_APP}")
else:
    print("\n❌ Projekt wurde wegen Fehlern abgebrochen.")
    print("ℹ️  Bitte beheben Sie die Fehler und führen Sie das Skript erneut aus.")

print("\n🏁 Projekt-Ausführung beendet.")