#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/__init__.py - Point d'entrée centralisé du package ML
"""

import warnings
import sys
import os
from pathlib import Path

# Version du package
__version__ = '1.0.0'

# Ajouter le répertoire parent au path pour les imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Stockage des imports réussis/échoués
_imported_modules = {}
_failed_imports = {}
_all_imports = {}

def _safe_import_module(module_name, class_names):
    """Import sécurisé d'un module avec gestion d'erreurs."""
    imported_items = {}
    
    try:
        # Essayer d'importer le module depuis le répertoire courant
        module = __import__(module_name, fromlist=class_names)
        
        # Import des classes/fonctions spécifiques
        for class_name in class_names:
            if hasattr(module, class_name):
                imported_items[class_name] = getattr(module, class_name)
            else:
                warnings.warn(f"Classe/fonction '{class_name}' non trouvée dans le module '{module_name}'")
        
        if imported_items:
            _imported_modules[module_name] = imported_items
            return imported_items
        
    except ImportError as e:
        error_msg = f"Import du module '{module_name}' échoué: {str(e)}"
        _failed_imports[module_name] = str(e)
        # Ne pas faire de warning pour des imports optionnels
        if module_name in ['config']:  # modules critiques seulement
            warnings.warn(error_msg)
        
    except Exception as e:
        error_msg = f"Erreur lors de l'import du module '{module_name}': {str(e)}"
        _failed_imports[module_name] = str(e)
        warnings.warn(error_msg)
    
    return {}

# 1. Import prioritaire : Configuration
try:
    config_imports = _safe_import_module('config', [
        'Config', 'LoggerConfig', 'ProjectConfig', 
        'get_config', 'get_logger', 'get_project_config'
    ])
    _all_imports.update(config_imports)
except:
    # Fallback si config n'est pas disponible
    warnings.warn("Module config non disponible - fonctionnalités limitées")

# 2. Modules de base
modules_to_import = {
    'data_extractor': ['DataExtractor', 'extract_data'],
    'dimension_optimizer': ['DimensionOptimizer', 'DataPipeline', 'run_pipeline'],
    'age_clustering': ['AgeClusterer', 'run_full_pipeline'],
    'client_scoring': ['ClientScoringEngine', 'quick_client_scoring'],
    'advanced_pipeline': ['AdvancedBankingPipeline', 'run_advanced_pipeline']
}

for module_name, class_names in modules_to_import.items():
    try:
        module_imports = _safe_import_module(module_name, class_names)
        _all_imports.update(module_imports)
    except:
        continue

# Export des éléments importés avec succès dans le namespace global
globals().update(_all_imports)

# Liste dynamique des éléments exportés
__all__ = list(_all_imports.keys())

# Fonctions utilitaires pour diagnostic
def get_import_status():
    """Retourne le statut des imports."""
    return {
        'imported_modules': list(_imported_modules.keys()),
        'failed_imports': _failed_imports,
        'available_classes': list(_all_imports.keys()),
        'total_imported': len(_all_imports),
        'version': __version__
    }

def print_import_status():
    """Affiche le statut des imports."""
    status = get_import_status()
    
    print(f"=== Status des imports - Package ML v{status['version']} ===")
    print(f"Modules importés avec succès: {len(status['imported_modules'])}")
    for module in status['imported_modules']:
        print(f"  ✓ {module}")
    
    if status['failed_imports']:
        print(f"\nModules échoués: {len(status['failed_imports'])}")
        for module, error in status['failed_imports'].items():
            print(f"  ✗ {module}: {error}")
    
    print(f"\nClasses/fonctions disponibles: {status['total_imported']}")
    for name in sorted(status['available_classes']):
        print(f"  - {name}")

def check_dependencies():
    """Vérifie si les dépendances principales sont satisfaites."""
    # Au minimum, on devrait avoir quelques classes de base
    critical_classes = ['DataExtractor', 'DimensionOptimizer']
    
    missing = []
    for cls in critical_classes:
        if cls not in _all_imports:
            missing.append(cls)
    
    if missing:
        warnings.warn(f"Classes critiques manquantes: {missing}")
        return False
    
    return True

# Message d'information simplifié
if len(_all_imports) == 0:
    warnings.warn("Aucun module n'a pu être importé - vérifiez la structure du projet")
elif len(_failed_imports) > len(_imported_modules):
    warnings.warn(f"Plusieurs imports ont échoué. Utilisez print_import_status() pour plus de détails.")

# Auto-diagnostic si disponible
if 'print_import_status' in globals() and os.getenv('DEBUG_IMPORTS', '').lower() == 'true':
    print_import_status()
