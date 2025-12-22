#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py - Configuration centralisée du projet ML
Version réécrite avec logging fonctionnel garanti
"""

import os
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime


class LoggerConfig:
    """Configuration du logging simplifiée et robuste."""
    
    # Configuration par défaut
    DEFAULT_CONFIG = {
        'level': logging.INFO,
        'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        'date_format': "%Y-%m-%d %H:%M:%S",
        'console_enabled': True,
        'file_enabled': True,
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }
    
    @classmethod
    def create_logger(cls, name: str, log_file_path: str = None) -> logging.Logger:
        """
        Crée un logger avec garantie d'écriture dans un fichier.
        
        Args:
            name: Nom du logger
            log_file_path: Chemin du fichier de log (optionnel)
            
        Returns:
            Logger configuré
        """
        logger = logging.getLogger(name)
        logger.setLevel(cls.DEFAULT_CONFIG['level'])
        
        # Si le logger a déjà des handlers, on le retourne tel quel
        if logger.handlers:
            return logger
        
        # Formatter commun
        formatter = logging.Formatter(
            cls.DEFAULT_CONFIG['format'], 
            cls.DEFAULT_CONFIG['date_format']
        )
        
        # 1. Handler console (toujours activé)
        if cls.DEFAULT_CONFIG['console_enabled']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(cls.DEFAULT_CONFIG['level'])
            logger.addHandler(console_handler)
        
        # 2. Handler fichier (si demandé)
        if cls.DEFAULT_CONFIG['file_enabled'] and log_file_path:
            try:
                # S'assurer que le répertoire existe
                log_path = Path(log_file_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Créer le handler fichier avec rotation
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    str(log_path),
                    maxBytes=cls.DEFAULT_CONFIG['max_file_size'],
                    backupCount=cls.DEFAULT_CONFIG['backup_count'],
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(cls.DEFAULT_CONFIG['level'])
                logger.addHandler(file_handler)
                
                # Message de confirmation
                logger.info(f"Logger {name} configuré - Fichier: {log_path}")
                
            except Exception as e:
                # Si la création du handler fichier échoue, on continue avec console seulement
                logger.warning(f"Impossible de créer le fichier de log {log_file_path}: {e}")
                logger.warning("Logging uniquement en console")
        
        return logger


def _find_project_root() -> Path:
    """
    Trouve automatiquement la racine du projet.
    Cherche le répertoire contenant 'src' ou utilise le répertoire courant.
    """
    current = Path.cwd()
    
    # Si on est déjà dans un répertoire avec src/, on remonte
    if (current / "src").exists():
        return current
    
    # Si on est dans src/, on remonte d'un niveau
    if current.name == "src" and current.parent.exists():
        return current.parent
    
    # Sinon, on utilise le répertoire courant
    return current


class ProjectConfig:
    """Configuration centralisée du projet avec gestion des chemins."""
    
    # === CHEMINS DE BASE ===
    PROJECT_ROOT = _find_project_root()
    SRC_DIR = PROJECT_ROOT / "src"
    MODELS_DIR = PROJECT_ROOT / "models"
    CACHE_DIR = PROJECT_ROOT / "cache"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # === BASE DE DONNÉES ===
    DATABASE_NAME = os.getenv("DATABASE_NAME", "DEFAULT_DB")
    
    # === CACHE ===
    CACHE_FILES = {
        'core_data': 'core_data.pkl',
        'organizational_data': 'org_data.pkl', 
        'financial_data': 'financial_data.pkl',
        'processed_data': 'processed_data.pkl',
        'clustering_models': 'clustering_models.pkl',
        'scoring_models': 'scoring_models.pkl'
    }
    
    # === EXTRACTION ===
    EXTRACTION = {
        'id_column': 'ID_DWR_CLI_CIAL',
        'age_filters': {'min': 18, 'max': 100},
        'age_min': 18,
        'age_max': 100,
        'nb_majeur_min': 1,
        'default_sample_size': 10000
    }
    
    # === CLUSTERING ===
    CLUSTERING = {
        'age_bins': 5,
        'k_range': (2, 6),
        'min_cluster_size': 800,
        'min_strata_size': 10000,
        'random_state': 42
    }
    
    # === SCORING ===
    SCORING = {
        'top_client_pct': 0.25,
        'pnb_columns': ['PNB_COLL', 'PNB_CRED', 'PNB_SERV', 'PNB_ASSU']
    }
    
    # === LOGGING ===
    LOGGING = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'max_file_size': 10 * 1024 * 1024,
        'backup_count': 5
    }
    
    @classmethod
    def ensure_directories(cls) -> bool:
        """
        Crée tous les répertoires nécessaires.
        
        Returns:
            True si tous les répertoires ont été créés, False sinon
        """
        directories = [cls.SRC_DIR, cls.MODELS_DIR, 
                      cls.CACHE_DIR, cls.LOGS_DIR]
        
        success = True
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Vérifier les permissions d'écriture
                if not os.access(directory, os.W_OK):
                    print(f"⚠️  ATTENTION: Pas de permission d'écriture dans {directory}")
            except Exception as e:
                print(f"❌ ERREUR: Impossible de créer {directory}: {e}")
                success = False
        
        return success
    
    @classmethod
    def get_cache_path(cls, cache_type: str) -> Path:
        """Retourne le chemin complet d'un fichier de cache."""
        if cache_type not in cls.CACHE_FILES:
            raise ValueError(f"Type de cache inconnu: {cache_type}. Types disponibles: {list(cls.CACHE_FILES.keys())}")
        
        cls.ensure_directories()
        return cls.CACHE_DIR / cls.CACHE_FILES[cache_type]
    
    @classmethod
    def get_log_path(cls, module_name: str, include_date: bool = True) -> Path:
        """
        Retourne le chemin du fichier de log pour un module.
        
        Args:
            module_name: Nom du module
            include_date: Si True, inclut la date dans le nom du fichier
            
        Returns:
            Chemin du fichier de log
        """
        cls.ensure_directories()
        
        if include_date:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{module_name}_{timestamp}.log"
        else:
            filename = f"{module_name}.log"
        
        return cls.LOGS_DIR / filename


class Config:
    """Configuration principale - Interface simplifiée."""
    
    # Références vers les configurations
    PROJECT_ROOT = ProjectConfig.PROJECT_ROOT
    MODELS_DIR = ProjectConfig.MODELS_DIR
    CACHE_DIR = ProjectConfig.CACHE_DIR
    LOGS_DIR = ProjectConfig.LOGS_DIR
    
    # Configurations métier
    EXTRACTION = ProjectConfig.EXTRACTION
    CLUSTERING = ProjectConfig.CLUSTERING
    SCORING = ProjectConfig.SCORING
    LOGGING = ProjectConfig.LOGGING
    DATABASE = ProjectConfig.DATABASE_NAME
    
    @staticmethod
    def get_cache_path(cache_type: str) -> Path:
        """Obtenir le chemin d'un fichier de cache."""
        return ProjectConfig.get_cache_path(cache_type)
    
    @staticmethod
    def get_log_path(module_name: str, include_date: bool = True) -> Path:
        """Obtenir le chemin d'un fichier de log."""
        return ProjectConfig.get_log_path(module_name, include_date)


# === FONCTIONS PRINCIPALES ===

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Crée et configure un logger avec fichier de log garanti.
    
    Args:
        name: Nom du logger (utilise __name__ par défaut)
        
    Returns:
        Logger configuré avec console + fichier
        
    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Message de test")
        
        >>> logger = get_logger("mon_module")
        >>> logger.error("Erreur dans mon_module")
    """
    # Extraire le nom du module (dernier élément après le point)
    module_name = name.split('.')[-1] if '.' in name else name
    
    # Générer le chemin du fichier de log
    log_file = Config.get_log_path(module_name)
    
    # Créer le logger
    logger = LoggerConfig.create_logger(name, str(log_file))
    
    return logger


def get_simple_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Version simplifiée pour créer un logger rapidement.
    
    Args:
        name: Nom du logger
        log_to_file: Si False, log seulement en console
        
    Returns:
        Logger configuré
    """
    if log_to_file:
        return get_logger(name)
    else:
        return LoggerConfig.create_logger(name, None)


def list_log_files() -> List[Path]:
    """
    Liste tous les fichiers de log existants.
    
    Returns:
        Liste des chemins des fichiers de log, triés par date de modification
    """
    logs_dir = Config.LOGS_DIR
    
    if not logs_dir.exists():
        print(f"📁 Le répertoire de logs {logs_dir} n'existe pas encore")
        return []
    
    log_files = list(logs_dir.glob("*.log"))
    
    if not log_files:
        print(f"📁 Aucun fichier de log dans {logs_dir}")
        return []
    
    # Trier par date de modification (plus récent en premier)
    log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    print(f"📋 Fichiers de log dans {logs_dir}:")
    for log_file in log_files:
        size = log_file.stat().st_size
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        print(f"   📄 {log_file.name} ({size:,} bytes, modifié: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return log_files


def show_recent_logs(n: int = 3):
    """
    Affiche le contenu des n fichiers de log les plus récents.
    
    Args:
        n: Nombre de fichiers à afficher
    """
    log_files = list_log_files()
    
    if not log_files:
        return
    
    print(f"\n📖 Contenu des {min(n, len(log_files))} fichiers de log les plus récents:")
    
    for i, log_file in enumerate(log_files[:n]):
        print(f"\n{'='*60}")
        print(f"📄 {log_file.name}")
        print(f"{'='*60}")
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:
                # Afficher les dernières lignes si le fichier est très long
                lines = content.split('\n')
                if len(lines) > 20:
                    print("... (fichier tronqué, dernières 20 lignes)")
                    print('\n'.join(lines[-20:]))
                else:
                    print(content)
            else:
                print("(Fichier vide)")
                
        except Exception as e:
            print(f"❌ Erreur lecture: {e}")


def test_logging():
    """
    Fonction de test pour vérifier que le logging fonctionne.
    """
    print("🧪 TEST DU SYSTÈME DE LOGGING")
    print("="*50)
    
    # Test avec différents modules
    test_modules = ["test_module", "main", "data_processor", "model_trainer"]
    
    print("📝 Création de loggers de test...")
    for module in test_modules:
        logger = get_logger(module)
        logger.info(f"Message de test INFO pour {module}")
        logger.warning(f"Message de test WARNING pour {module}")
        logger.error(f"Message de test ERROR pour {module}")
    
    print("\n✅ Messages envoyés. Vérification des fichiers créés...\n")
    
    # Lister les fichiers créés
    log_files = list_log_files()
    
    # Afficher le contenu du dernier fichier
    if log_files:
        print(f"\n📖 Contenu du fichier le plus récent ({log_files[0].name}):")
        print("-" * 50)
        try:
            with open(log_files[0], 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print(f"❌ Erreur lecture: {e}")
    
    print("\n🎉 Test terminé!")


def get_config():
    """Retourne la configuration par défaut."""
    return Config()


def get_project_config():
    """Retourne la configuration projet par défaut."""
    return ProjectConfig()


# === INSTANCES PAR DÉFAUT ===
DEFAULT_PROJECT_CONFIG = ProjectConfig()
DEFAULT_CONFIG = Config()


# === INITIALISATION ===
def _initialize_config():
    """Initialise la configuration au chargement du module."""
    print("🚀 Initialisation de la configuration...")
    print(f"📁 Racine du projet: {ProjectConfig.PROJECT_ROOT}")
    print(f"📁 Répertoire de logs: {ProjectConfig.LOGS_DIR}")
    
    try:
        success = ProjectConfig.ensure_directories()
        if success:
            print("✅ Tous les répertoires ont été créés/vérifiés")
        else:
            print("⚠️  Certains répertoires n'ont pas pu être créés")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")

# === TEST PRINCIPAL ===
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 EXÉCUTION DU TEST DE CONFIGURATION")
    print("="*60)
    
    # Test complet
    test_logging()
    
    # Affichage des logs récents
    show_recent_logs(2)
    
    print("\n✨ Configuration testée avec succès!")
