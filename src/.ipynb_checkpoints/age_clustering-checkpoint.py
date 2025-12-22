#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
age_clustering.py - Module de clustering stratifié par âge

Clustering par strates d'âge avec persistance des modèles et monitoring avancé.
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from .config import Config, get_logger


class AgeClusterer:
    """
    Clustering stratifié par âge avec persistance des modèles et monitoring.
    
    Fonctionnalités :
    - Strates fixes de 5 ans (18-22, 23-27, ..., 63+)
    - K-means adaptatif par strate avec contraintes métier
    - Sauvegarde/chargement des modèles dans models/
    - Monitoring avancé pour alertes de recalcul
    - Historique des performances
    """
    
    def __init__(self, model_version: str = None):
        """
        Initialise le clustering stratifié.
        
        Args:
            model_version: Version du modèle (auto si None)
        """
        
        # Configuration depuis Config centralisé
        self.strata_size = Config.CLUSTERING['age_bins']
        self.k_range = Config.CLUSTERING['k_range']
        self.min_cluster_size = Config.CLUSTERING['min_cluster_size']
        self.min_strata_size = Config.CLUSTERING['min_strata_size']
        self.random_state = Config.CLUSTERING['random_state']
        
        # Version du modèle
        if model_version is None:
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.model_version = model_version
        
        # Chemins de sauvegarde
        self.model_path = Config.MODELS_DIR / f"age_clustering_{self.model_version}.pkl"
        self.history_path = Config.MODELS_DIR / "clustering_history.json"
        
        # Logger unifié
        self.logger = get_logger('age_clustering')
        
        # Stockage des modèles et résultats
        self.age_strata = {}
        self.scaler_models = {}
        self.kmeans_models = {}
        self.quality_metrics = {}
        self.training_metadata = {}
        self.is_fitted = False
        
        self.logger.info(f"AgeClusterer initialisé - Version: {self.model_version}")
        self.logger.info(f"Configuration - Strates: {self.strata_size} ans, K-range: {self.k_range}")
        self.logger.info(f"Contraintes - Min cluster: {self.min_cluster_size}, Min strate: {self.min_strata_size}")
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Entraîne les modèles et applique le clustering stratifié.
        
        Args:
            df: DataFrame avec dimensions réduites et colonne AGE
            
        Returns:
            DataFrame avec colonnes de clustering ajoutées
        """
        
        self.logger.info("=== DÉBUT ENTRAÎNEMENT CLUSTERING STRATIFIÉ ===")
        self.logger.info(f"Version modèle: {self.model_version}")
        self.logger.info(f"Données d'entrée : {df.shape}")
        
        start_time = time.time()
        
        # Validation des données
        self._validate_input_data(df)
        
        # Préparation
        feature_cols = self._get_feature_columns(df)
        self.logger.info(f"Features utilisées : {len(feature_cols)} : {feature_cols}")
        
        # Création des strates fixes
        self.age_strata = self._create_age_strata()
        self._log_strata_distribution(df)
        
        # Clustering par strate
        cluster_results = self._cluster_all_strata(df, feature_cols)
        
        # Consolidation des résultats
        df_result = self._consolidate_results(df, cluster_results)
        
        # Calcul des métriques détaillées
        self._calculate_detailed_metrics(cluster_results, df)
        
        # Sauvegarde du modèle
        self._save_model()
        
        # Enregistrement dans l'historique
        self._update_history()
        
        # Log des alertes et recommandations
        self._log_model_alerts()
        
        # Marquage comme entraîné
        self.is_fitted = True
        
        total_time = time.time() - start_time
        total_clusters = sum(len(result['cluster_sizes']) for result in cluster_results.values())
        
        self.logger.info("=== ENTRAÎNEMENT TERMINÉ ===")
        self.logger.info(f"Durée: {total_time:.2f}s - {total_clusters} clusters créés")
        self.logger.info(f"Modèle sauvegardé: {self.model_path}")
        
        return df_result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique le clustering à de nouvelles données"""
        
        if not self.is_fitted:
            self.logger.error("Modèle non entraîné - tentative de chargement automatique")
            latest_model = self._find_latest_model()
            if latest_model:
                self.load_model(latest_model)
            else:
                raise ValueError("Aucun modèle disponible")
        
        self.logger.info("=== APPLICATION DU CLUSTERING ===")
        self.logger.info(f"Nouvelles données : {df.shape}")
        
        start_time = time.time()
        
        # Validation
        self._validate_input_data(df)
        feature_cols = self._get_feature_columns(df)
        
        df_result = df.copy()
        df_result['CLUSTER_STRATE'] = 'NON_CLUSTERED'
        df_result['CLUSTER_LOCAL'] = -1
        df_result['CLUSTER_GLOBAL'] = 'NON_CLUSTERED'
        
        # Application par strate avec monitoring
        assigned_clients = 0
        strata_performance = {}
        
        for strata_name, (age_min, age_max) in self.age_strata.items():
            if strata_name not in self.scaler_models:
                continue
            
            # Filtrage
            if age_max == 999: 
                strata_mask = (df['AGE'] >= age_min)
            else:
                strata_mask = (df['AGE'] >= age_min) & (df['AGE'] <= age_max)
                
            df_strata = df[strata_mask]
            
            if len(df_strata) == 0:
                continue
            
            # Application des modèles avec monitoring
            try:
                X = df_strata[feature_cols].fillna(0)
                X_scaled = self.scaler_models[strata_name].transform(X)
                
                # Prédiction selon le type de modèle
                if strata_name in self.kmeans_models:
                    # Modèle k-means normal
                    local_labels = self.kmeans_models[strata_name].predict(X_scaled)
                    
                    # Calcul silhouette pour monitoring
                    if len(np.unique(local_labels)) > 1:
                        silhouette_new = silhouette_score(X_scaled, local_labels)
                        strata_performance[strata_name] = {
                            'silhouette_new': silhouette_new,
                            'silhouette_training': self.quality_metrics.get('strata_details', {}).get(strata_name, {}).get('silhouette_score', 0),
                            'n_clients': len(df_strata)
                        }
                else:
                    # Cluster unique - tous assignés à 0
                    local_labels = np.zeros(len(df_strata), dtype=int)
                    strata_performance[strata_name] = {
                        'silhouette_new': 0.0,
                        'silhouette_training': 0.0,
                        'n_clients': len(df_strata)
                    }
                
                # Assignation
                global_labels = [f"{strata_name}_C{label}" for label in local_labels]
                
                df_result.loc[df_strata.index, 'CLUSTER_STRATE'] = strata_name
                df_result.loc[df_strata.index, 'CLUSTER_LOCAL'] = local_labels
                df_result.loc[df_strata.index, 'CLUSTER_GLOBAL'] = global_labels
                
                assigned_clients += len(df_strata)
                
            except Exception as e:
                self.logger.error(f"Erreur application strate {strata_name}: {e}")
        
        # Log monitoring des performances
        self._log_application_performance(strata_performance)
        
        total_time = time.time() - start_time
        self.logger.info(f"Application terminée en {total_time:.2f}s - {assigned_clients}/{len(df)} clients assignés")
        
        return df_result
    
    def load_model(self, model_path: str = None):
        """Charge un modèle précédemment sauvegardé"""
        
        if model_path is None:
            model_path = self._find_latest_model()
            if model_path is None:
                raise ValueError("Aucun modèle trouvé")
        
        model_path = Path(model_path)
        self.logger.info(f"Chargement modèle depuis : {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restauration
            self.age_strata = model_data['age_strata']
            self.scaler_models = model_data['scaler_models']
            self.kmeans_models = model_data['kmeans_models']
            self.quality_metrics = model_data['quality_metrics']
            self.training_metadata = model_data['training_metadata']
            self.model_version = model_data.get('model_version', 'unknown')
            
            self.is_fitted = True
            
            self.logger.info("=== MODÈLE CHARGÉ AVEC SUCCÈS ===")
            self.logger.info(f"Version: {self.model_version}")
            self.logger.info(f"Date entraînement: {self.training_metadata.get('training_date', 'inconnue')}")
            self.logger.info(f"Nb échantillons: {self.training_metadata.get('n_samples', 'inconnu')}")
            self.logger.info(f"Score silhouette moyen: {self.quality_metrics.get('global_metrics', {}).get('avg_silhouette_score', 'inconnu'):.3f}")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle: {e}")
            raise
    
    def _validate_input_data(self, df: pd.DataFrame):
        """Valide les données d'entrée avec logging détaillé"""
        if df.empty:
            self.logger.error("DataFrame vide")
            raise ValueError("DataFrame vide")
        
        if 'AGE' not in df.columns:
            self.logger.error("Colonne 'AGE' manquante")
            raise ValueError("Colonne 'AGE' manquante")
        
        # Analyse de la qualité des âges
        age_numeric = pd.to_numeric(df['AGE'], errors='coerce')
        invalid_ages = age_numeric.isna().sum()
        
        if invalid_ages > 0:
            self.logger.warning(f"QUALITÉ DONNÉE - {invalid_ages} âges invalides ({invalid_ages/len(df)*100:.1f}%)")
        
        valid_ages = age_numeric.dropna()
        if len(valid_ages) == 0:
            self.logger.error("Aucun âge valide trouvé")
            raise ValueError("Aucun âge valide")
        
        age_min, age_max = valid_ages.min(), valid_ages.max()
        age_mean, age_std = valid_ages.mean(), valid_ages.std()
        
        self.logger.info(f"PROFIL DONNÉE - Âges: {age_min:.0f} à {age_max:.0f} ans (μ={age_mean:.1f}±{age_std:.1f})")
        
        # Alertes de dérive
        if hasattr(self, 'training_metadata') and self.training_metadata:
            prev_mean = self.training_metadata.get('age_mean')
            prev_std = self.training_metadata.get('age_std')
            if prev_mean and abs(age_mean - prev_mean) > 2:
                self.logger.warning(f"ALERTE DÉRIVE - Âge moyen dérivé: {prev_mean:.1f} → {age_mean:.1f}")
            if prev_std and abs(age_std - prev_std) > 1:
                self.logger.warning(f"ALERTE DÉRIVE - Écart-type âge dérivé: {prev_std:.1f} → {age_std:.1f}")
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identifie les colonnes de features pour le clustering"""
        excluded_cols = [Config.EXTRACTION['id_column'], 'CT_CLI_CIAL', 'AGE']
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        
        if not feature_cols:
            self.logger.error("Aucune colonne de feature trouvée")
            raise ValueError("Aucune colonne de feature")
        
        return feature_cols
    
    def _create_age_strata(self) -> Dict[str, Tuple[int, int]]:
        """Crée les strates d'âge fixes : 18-22, 23-27, ..., 63+"""
        strata = {}
        
        # Tranches de 5 ans de 18 à 63
        current_age = 18
        while current_age <= 59:
            next_age = current_age + 4  # 5 ans : 18-22, 23-27, etc.
            strata_name = f"{current_age:02d}-{next_age:02d}"
            strata[strata_name] = (current_age, next_age)
            current_age = next_age + 1
        
        strata["63+"] = (63, 999)
        
        return strata
    
    def _log_strata_distribution(self, df: pd.DataFrame):
        """Log détaillé de la distribution par strate"""
        self.logger.info("=== DISTRIBUTION PAR STRATE ===")
        
        strata_stats = {}
        for strata_name, (age_min, age_max) in self.age_strata.items():
            if age_max == 999:  
                strata_count = (df['AGE'] >= age_min).sum()
                strata_stats[strata_name] = strata_count
                self.logger.info(f"  {strata_name} ({age_min}+): {strata_count:,} clients")
            else:
                strata_count = ((df['AGE'] >= age_min) & (df['AGE'] <= age_max)).sum()
                strata_stats[strata_name] = strata_count
                self.logger.info(f"  {strata_name} ({age_min}-{age_max}): {strata_count:,} clients")
        
        # Identification des strates sous-représentées
        under_threshold = [name for name, count in strata_stats.items() 
                          if count < self.min_strata_size]
        if under_threshold:
            self.logger.warning(f"ALERTE STRATE - Strates sous-représentées: {under_threshold}")
    
    def _cluster_all_strata(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Effectue le clustering pour toutes les strates"""
        cluster_results = {}
        
        for strata_name, (age_min, age_max) in self.age_strata.items():
            self.logger.info(f"--- CLUSTERING STRATE {strata_name} ---")
            
            # Filtrage de la strate
            if age_max == 999:
                strata_mask = (df['AGE'] >= age_min)
            else:
                strata_mask = (df['AGE'] >= age_min) & (df['AGE'] <= age_max)
            
            df_strata = df[strata_mask].copy()
            
            if len(df_strata) < self.min_strata_size:
                self.logger.warning(f"STRATE IGNORÉE - {strata_name}: {len(df_strata)} < {self.min_strata_size}")
                continue
            
            # Clustering de la strate
            strata_result = self._cluster_single_strata(df_strata, feature_cols, strata_name)
            
            if strata_result is not None:
                cluster_results[strata_name] = strata_result
                self.logger.info(f"RÉSULTAT - {strata_name}: {strata_result['optimal_k']} clusters, Silhouette={strata_result['silhouette_score']:.3f}")
            else:
                self.logger.error(f"ÉCHEC CLUSTERING - {strata_name}")
        
        return cluster_results
    
    def _cluster_single_strata(self, df_strata: pd.DataFrame, feature_cols: List[str], strata_name: str) -> Optional[Dict[str, Any]]:
        """Clustering pour une strate spécifique avec monitoring détaillé"""
        try:
            # Préparation des features
            X = df_strata[feature_cols].copy()
            X = X.fillna(0)  # Gestion des valeurs manquantes
            
            # Standardisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Recherche du K optimal
            best_result = self._find_optimal_k(X_scaled, len(df_strata), strata_name)
            
            if best_result is None:
                return None
            
            # Clustering final
            optimal_k = best_result['k']
            
            if optimal_k == 1:
                # Cas spécial : cluster unique
                cluster_labels = np.zeros(len(df_strata), dtype=int)
                silhouette_avg = 0.0
                self.logger.info(f"CLUSTER UNIQUE - {strata_name}")
            else:
                kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                # Sauvegarde du modèle k-means
                self.kmeans_models[strata_name] = kmeans
            
            # Métriques finales
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            
            # Sauvegarde du scaler
            self.scaler_models[strata_name] = scaler
            
            return {
                'indices': df_strata.index,
                'cluster_labels': pd.Series(cluster_labels, index=df_strata.index),
                'silhouette_score': silhouette_avg,
                'cluster_sizes': cluster_sizes.to_dict(),
                'optimal_k': optimal_k,
                'n_samples': len(df_strata),
                'feature_importance': self._calculate_feature_importance(X_scaled, cluster_labels, feature_cols)
            }
            
        except Exception as e:
            self.logger.error(f"ERREUR CLUSTERING - {strata_name}: {e}")
            return None
    
    def _find_optimal_k(self, X_scaled: np.ndarray, strata_size: int, strata_name: str) -> Optional[Dict[str, Any]]:
        """Trouve le K optimal avec logging détaillé"""
        max_k = min(self.k_range[1], strata_size // self.min_cluster_size)
        
        if max_k < self.k_range[0]:
            self.logger.warning(f"K CONTRAINT - {strata_name}: max_k ({max_k}) < min_k ({self.k_range[0]})")
            max_k = 2
        
        best_k = None
        best_silhouette = -1
        k_results = {}
        
        # Test des K possibles
        for k in range(max(2, self.k_range[0]), max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                # Vérification contrainte taille
                cluster_sizes = pd.Series(labels).value_counts()
                min_size = cluster_sizes.min()
                
                if min_size < self.min_cluster_size:
                    k_results[k] = {'silhouette': 'INVALID', 'min_size': min_size}
                    continue
                
                # Score silhouette
                silhouette_avg = silhouette_score(X_scaled, labels)
                k_results[k] = {'silhouette': silhouette_avg, 'min_size': min_size}
                
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_k = k
                    
            except Exception as e:
                k_results[k] = {'silhouette': f'ERROR: {e}', 'min_size': 0}
        
        # Log des résultats K avec détails
        self.logger.info(f"OPTIMISATION K - {strata_name}:")
        for k, result in k_results.items():
            self.logger.info(f"  K={k}: Silhouette={result['silhouette']}, MinSize={result['min_size']}")
        
        # Fallback à K=1 si aucun K valide trouvé
        if best_k is None:
            self.logger.warning(f"FALLBACK K=1 - {strata_name}: aucun K valide")
            return {'k': 1, 'silhouette': 0.0}
        
        self.logger.info(f"K OPTIMAL - {strata_name}: K={best_k} (Silhouette={best_silhouette:.3f})")
        return {'k': best_k, 'silhouette': best_silhouette}
    
    def _calculate_feature_importance(self, X_scaled: np.ndarray, labels: np.ndarray, feature_cols: List[str]) -> Dict[str, float]:
        """Calcule l'importance des features par strate"""
        try:
            # Cas spécial : cluster unique
            if len(np.unique(labels)) == 1:
                return {col: 0.0 for col in feature_cols}
            
            # Centroïdes des clusters
            centroids = []
            for cluster_id in np.unique(labels):
                cluster_data = X_scaled[labels == cluster_id]
                centroids.append(np.mean(cluster_data, axis=0))
            
            centroids = np.array(centroids)
            
            # Variance inter-clusters par feature
            feature_importance = {}
            for i, feature in enumerate(feature_cols):
                feature_variance = np.var(centroids[:, i])
                feature_importance[feature] = float(feature_variance)
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"ERREUR FEATURE IMPORTANCE: {e}")
            return {col: 0.0 for col in feature_cols}
    
    def _consolidate_results(self, df_original: pd.DataFrame, cluster_results: Dict) -> pd.DataFrame:
        """Consolide les résultats avec assignation globale"""
        df_result = df_original.copy()
        
        # Initialisation
        df_result['CLUSTER_STRATE'] = 'NON_CLUSTERED'
        df_result['CLUSTER_LOCAL'] = -1
        df_result['CLUSTER_GLOBAL'] = 'NON_CLUSTERED'
        
        # Assignation par strate
        for strata_name, strata_data in cluster_results.items():
            indices = strata_data['indices']
            local_labels = strata_data['cluster_labels']
            
            # Mapping global
            global_labels = [f"{strata_name}_C{label}" for label in local_labels]
            
            df_result.loc[indices, 'CLUSTER_STRATE'] = strata_name
            df_result.loc[indices, 'CLUSTER_LOCAL'] = local_labels
            df_result.loc[indices, 'CLUSTER_GLOBAL'] = global_labels
        
        return df_result
    
    def _calculate_detailed_metrics(self, cluster_results: Dict, df: pd.DataFrame):
        """Calcule les métriques détaillées avec métadonnées d'entraînement"""
        if not cluster_results:
            self.quality_metrics = {'error': 'Aucun clustering réussi'}
            return
        
        # Métadonnées d'entraînement
        self.training_metadata = {
            'training_date': datetime.now().isoformat(),
            'model_version': self.model_version,
            'n_samples': len(df),
            'n_features': len(self._get_feature_columns(df)),
            'age_mean': float(df['AGE'].mean()),
            'age_std': float(df['AGE'].std()),
            'age_min': int(df['AGE'].min()),
            'age_max': int(df['AGE'].max())
        }
        
        # Agrégation des métriques
        all_silhouette = []
        total_clients = 0
        total_clusters = 0
        strata_details = {}
        feature_importance_global = {}
        single_cluster_strata = 0
        
        for strata_name, strata_data in cluster_results.items():
            if strata_data['silhouette_score'] > 0:
                all_silhouette.append(strata_data['silhouette_score'])
            else:
                single_cluster_strata += 1
                
            total_clients += strata_data['n_samples']
            total_clusters += strata_data['optimal_k']
            
            # Détails par strate
            strata_details[strata_name] = {
                'n_clients': strata_data['n_samples'],
                'n_clusters': strata_data['optimal_k'],
                'silhouette_score': strata_data['silhouette_score'],
                'cluster_sizes': strata_data['cluster_sizes'],
                'feature_importance': strata_data['feature_importance'],
                'is_single_cluster': strata_data['optimal_k'] == 1
            }
            
            # Agrégation importance des features
            for feature, importance in strata_data['feature_importance'].items():
                if feature not in feature_importance_global:
                    feature_importance_global[feature] = 0
                feature_importance_global[feature] += importance * strata_data['n_samples']
        
        # Normalisation de l'importance des features
        for feature in feature_importance_global:
            feature_importance_global[feature] /= total_clients
        
        # Métriques consolidées
        self.quality_metrics = {
            'timestamp': datetime.now().isoformat(),
            'global_metrics': {
                'total_clusters': total_clusters,
                'total_clustered_clients': total_clients,
                'viable_strata': len(cluster_results),
                'single_cluster_strata': single_cluster_strata,
                'avg_silhouette_score': np.mean(all_silhouette) if all_silhouette else 0.0,
                'min_silhouette_score': np.min(all_silhouette) if all_silhouette else 0.0,
                'max_silhouette_score': np.max(all_silhouette) if all_silhouette else 0.0,
                'silhouette_std': np.std(all_silhouette) if all_silhouette else 0.0,
                'feature_importance_global': feature_importance_global
            },
            'strata_details': strata_details
        }
    
    def _save_model(self):
        """Sauvegarde le modèle avec métadonnées complètes"""
        model_data = {
            'model_version': self.model_version,
            'age_strata': self.age_strata,
            'scaler_models': self.scaler_models,
            'kmeans_models': self.kmeans_models,
            'quality_metrics': self.quality_metrics,
            'training_metadata': self.training_metadata,
            'config': {
                'strata_size': self.strata_size,
                'k_range': self.k_range,
                'min_cluster_size': self.min_cluster_size,
                'min_strata_size': self.min_strata_size,
                'random_state': self.random_state
            }
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"MODÈLE SAUVEGARDÉ - {self.model_path}")
    
    def _update_history(self):
        """Met à jour l'historique des modèles"""
        history_entry = {
            'model_version': self.model_version,
            'training_date': self.training_metadata['training_date'],
            'n_samples': self.training_metadata['n_samples'],
            'avg_silhouette_score': self.quality_metrics['global_metrics']['avg_silhouette_score'],
            'total_clusters': self.quality_metrics['global_metrics']['total_clusters'],
            'viable_strata': self.quality_metrics['global_metrics']['viable_strata'],
            'model_path': str(self.model_path)
        }
        
        # Chargement historique existant
        history = []
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                self.logger.warning(f"Erreur lecture historique: {e}")
        
        # Ajout nouvelle entrée
        history.append(history_entry)
        
        # Limitation à 100 entrées max
        if len(history) > 100:
            history = history[-100:]
        
        # Sauvegarde
        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"HISTORIQUE MIS À JOUR - {len(history)} entrées")
    
    def _log_model_alerts(self):
        """Log les alertes et recommandations basées sur les métriques"""
        self.logger.info("=== ANALYSE QUALITÉ & ALERTES ===")
        
        gm = self.quality_metrics['global_metrics']
        avg_silhouette = gm['avg_silhouette_score']
        
        # Évaluation qualité
        if avg_silhouette > 0.7:
            quality_level = "EXCELLENTE"
            alert_level = "INFO"
        elif avg_silhouette > 0.5:
            quality_level = "BONNE"
            alert_level = "INFO"
        elif avg_silhouette > 0.3:
            quality_level = "ACCEPTABLE"
            alert_level = "WARNING"
        else:
            quality_level = "FAIBLE"
            alert_level = "CRITICAL"
        
        self.logger.info(f"QUALITÉ GLOBALE: {quality_level} (Silhouette={avg_silhouette:.3f})")
        
        if alert_level == "WARNING":
            self.logger.warning("ALERTE QUALITÉ - Performances modérées, surveiller l'évolution")
        elif alert_level == "CRITICAL":
            self.logger.error("ALERTE CRITIQUE - Performances faibles, recalcul recommandé")
        
        # Alertes spécifiques
        if gm['single_cluster_strata'] > len(self.age_strata) * 0.3:
            self.logger.warning(f"ALERTE STRATE - {gm['single_cluster_strata']} strates à cluster unique")
        
        if gm['silhouette_std'] > 0.2:
            self.logger.warning(f"ALERTE COHÉRENCE - Forte variation silhouette (σ={gm['silhouette_std']:.3f})")
        
        # Recommandations de recalcul
        self._log_recalculation_recommendations()
    
    def _log_recalculation_recommendations(self):
        """Log les recommandations de recalcul basées sur l'historique"""
        if not self.history_path.exists():
            return
        
        try:
            with open(self.history_path, 'r') as f:
                history = json.load(f)
            
            if len(history) < 2:
                return
            
            # Analyse tendance
            recent_scores = [entry['avg_silhouette_score'] for entry in history[-5:]]
            
            if len(recent_scores) >= 3:
                # Détection de dégradation
                if recent_scores[-1] < recent_scores[0] - 0.1:
                    self.logger.warning("RECOMMANDATION RECALCUL - Dégradation détectée sur 5 derniers modèles")
                
                # Détection de stabilité (variance faible)
                if np.std(recent_scores) < 0.02 and recent_scores[-1] > 0.6:
                    self.logger.info("MODÈLE STABLE - Performances constantes, recalcul non urgent")
        
        except Exception as e:
            self.logger.warning(f"Erreur analyse historique: {e}")
    
    def _log_application_performance(self, strata_performance: Dict):
        """Log les performances lors de l'application"""
        self.logger.info("=== MONITORING APPLICATION ===")
        
        drift_alerts = []
        
        for strata_name, perf in strata_performance.items():
            silhouette_new = perf['silhouette_new']
            silhouette_training = perf['silhouette_training']
            n_clients = perf['n_clients']
            
            if silhouette_training > 0:  # Éviter division par zéro
                drift = abs(silhouette_new - silhouette_training) / silhouette_training
                
                self.logger.info(f"{strata_name}: Silhouette {silhouette_training:.3f}→{silhouette_new:.3f} ({n_clients} clients)")
                
                if drift > 0.2:  # 20% de dérive
                    drift_alerts.append(f"{strata_name} (dérive: {drift:.1%})")
        
        if drift_alerts:
            self.logger.warning(f"ALERTE DÉRIVE DÉTECTÉE - Strates: {', '.join(drift_alerts)}")
            self.logger.warning("RECOMMANDATION - Considérer un recalcul du modèle")
    
    def _find_latest_model(self) -> Optional[str]:
        """Trouve le modèle le plus récent"""
        model_files = list(Config.MODELS_DIR.glob("age_clustering_*.pkl"))
        if not model_files:
            return None
        
        # Tri par date de modification
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        return str(latest_model)


# Fonctions utilitaires
def run_full_pipeline(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Fonction utilitaire pour exécuter clustering avec pipeline externe"""
    # Import tardif pour éviter circularité
    from .dimension_optimizer import DataPipeline
    
    logger = get_logger('clustering_pipeline')
    logger.info("Démarrage pipeline clustering complète")
    
    # Étape 1-2 : Extraction + Optimisation
    pipeline = DataPipeline(clustering_mode="stratifie")
    df_processed = pipeline.run_complete_pipeline(sample_size)
    
    if df_processed.empty:
        logger.error("Pipeline data vide - arrêt")
        return pd.DataFrame()
    
    # Étape 3 : Clustering
    clusterer = AgeClusterer()
    df_final = clusterer.fit_transform(df_processed)
    
    logger.info("Pipeline clustering complète terminée")
    return df_final
