import psutil
import time
import csv
import sys
from datetime import datetime

# Nom du processus à surveiller (tel que défini dans ton launch file: executable='rtabmap')
PROCESS_NAME = "rtabmap"

def get_process_by_name(name):
    """Retrouve le PID du processus via son nom."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # On cherche 'rtabmap' dans le nom (parfois c'est 'rtabmap' tout court)
            if name.lower() in proc.info['name'].lower():
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def main():
    print(f"En attente du lancement de {PROCESS_NAME}...")
    proc = None
    while proc is None:
        proc = get_process_by_name(PROCESS_NAME)
        time.sleep(1)
    
    print(f"Processus {PROCESS_NAME} trouvé (PID: {proc.pid}) ! Enregistrement...")

    # Création du fichier CSV
    filename = f"perf_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # En-têtes des colonnes
        writer.writerow(["Timestamp", "CPU_Percent", "RAM_MB", "RAM_Percent"])

        try:
            while psutil.pid_exists(proc.pid):
                # Récupération des stats
                # cpu_percent(interval=None) est non bloquant
                cpu = proc.cpu_percent(interval=None) 
                mem_info = proc.memory_info()
                ram_mb = mem_info.rss / (1024 * 1024) # Conversion en MB
                ram_percent = proc.memory_percent()

                # Écriture
                writer.writerow([datetime.now().strftime('%H:%M:%S'), cpu, round(ram_mb, 2), round(ram_percent, 2)])
                
                # Feedback console léger
                print(f"Rec: RAM={ram_mb:.1f} MB | CPU={cpu}%", end='\r')
                
                time.sleep(1.0) # Une mesure par seconde
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print(f"\nLe processus {PROCESS_NAME} s'est arrêté.")

    print(f"\nFini ! Données sauvegardées dans {filename}")

if __name__ == "__main__":
    main()
