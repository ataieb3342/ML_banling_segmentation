#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Ajout path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import pipeline
from src.advanced_pipeline import run_advanced_pipeline

def main():
    # Exécution pipeline
    results = run_advanced_pipeline(mode='prod', clustering_mode='stratifie')
    
    # Ajout à l'historique
    history_file = Path(__file__).parent / "logs/execution_history.json"
    
    history_entry = {
        'date': datetime.now().isoformat(),
        'status': results['metadata']['status'],
        'clients_scored': results['metrics'].get('scoring', {}).get('global_metrics', {}).get('total_clients', 0),
        'potentiel_total': results['metrics'].get('scoring', {}).get('global_metrics', {}).get('total_potential_value', 0),
        'duration': str(results['metadata'].get('duration', 'N/A'))
    }
    
    # Lecture historique existant
    history = []
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    
    # Ajout nouvelle entrée
    history.append(history_entry)
    
    # Limitation à 100 dernières exécutions
    if len(history) > 100:
        history = history[-100:]
    
    # Sauvegarde
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"Pipeline terminé - Status: {results['metadata']['status']}")
    return 0 if results['metadata']['status'] == 'success' else 1

if __name__ == "__main__":
    sys.exit(main())
