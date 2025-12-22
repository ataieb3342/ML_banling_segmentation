#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_pipeline.py - Pipeline avancé avec gestion intelligente des modèles
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

from .config import Config, get_logger
from .data_extractor import DataExtractor
from .dimension_optimizer import DimensionOptimizer, DataPipeline
from .age_clustering import AgeClusterer
from .client_scoring import ClientScoringEngine

try:
    from SessionSpark import spark, BaseDonnees
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    spark = None
    BaseDonnees = None


class AdvancedBankingPipeline:
    """
    Pipeline avancé avec gestion intelligente des modèles et historique.
    
    Modes disponibles:
    - dev: Exécution locale avec sauvegarde fichiers
    - prod: Exécution complète avec persistance en base
    - test: Exécution réduite avec échantillon
    
    NOUVELLE LOGIQUE MODÈLES:
    - Réutilise les modèles existants par défaut
    - Recalcule seulement si dégradation détectée ou demandé explicitement
    """
    
    # Configuration pour logs détaillés
    
    def __init__(self, mode: str = 'dev', clustering_mode: str = 'stratifie', 
                 force_retrain: bool = False):
        """
        Initialise le pipeline avancé.
        
        Args:
            mode: Mode d'exécution ('dev', 'prod', 'test')
            clustering_mode: Mode de clustering ('global', 'stratifie')
            force_retrain: Force le recalcul des modèles même s'ils existent
        """
        # Validation des modes
        valid_modes = ['dev', 'prod', 'test']
        valid_clustering_modes = ['global', 'stratifie']
        
        if mode not in valid_modes:
            raise ValueError(f"Mode invalide. Utilisez: {valid_modes}")
        if clustering_mode not in valid_clustering_modes:
            raise ValueError(f"Mode clustering invalide. Utilisez: {valid_clustering_modes}")
            
        self.mode = mode
        self.clustering_mode = clustering_mode
        self.force_retrain = force_retrain
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Logger
        self.logger = get_logger('advanced_pipeline')
        
        # Résultats
        self.results = {}
        self.metrics = {}
        self.execution_metadata = {
            'start_time': datetime.now(),
            'mode': mode,
            'clustering_mode': clustering_mode,
            'execution_id': self.execution_id,
            'force_retrain': force_retrain
        }
        
        self.logger.info(f"AdvancedPipeline initialisé - Mode: {mode}, Clustering: {clustering_mode}")
        self.logger.info(f"Force retrain: {force_retrain}")
   
    def run(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Exécute le pipeline complet avec gestion intelligente des modèles.
        
        Args:
            sample_size: Taille d'échantillon (surtout pour mode test)
            
        Returns:
            Dict avec résultats et métriques
        """
        self.logger.info("=== DÉMARRAGE PIPELINE AVANCÉ ===")
        
        try:
            # Étape 1: Extraction données
            df_raw = self._extract_data(sample_size)
            if df_raw.empty:
                raise ValueError("Échec extraction données")
            
            # Étape 2: Optimisation dimensions
            df_processed = self._optimize_dimensions(df_raw)
            if df_processed.empty:
                raise ValueError("Échec optimisation dimensions")
            
            # Étape 3: Clustering INTELLIGENT
            df_clustered = self._apply_smart_clustering(df_processed)
            if df_clustered.empty:
                raise ValueError("Échec clustering")
            
            # Étape 4: Scoring
            df_scored = self._apply_scoring(df_clustered, df_raw)
            if df_scored.empty:
                raise ValueError("Échec scoring")
            
            # Étape 5: Persistance (mode prod)
            if self.mode == 'prod':
                self._persist_results(df_scored)
            
            # Étape 6: Sauvegarde résultats (mode dev)
            if self.mode == 'dev':
                self._save_dev_results(df_scored)
            
            # Métadonnées d'exécution
            self.execution_metadata['end_time'] = datetime.now()
            self.execution_metadata['duration'] = (
                self.execution_metadata['end_time'] - 
                self.execution_metadata['start_time']
            )
            self.execution_metadata['status'] = 'success'
            
            self.logger.info("=== PIPELINE TERMINÉ AVEC SUCCÈS ===")
            
            return {
                'data': df_scored,
                'metrics': self.metrics,
                'metadata': self.execution_metadata
            }
            
        except Exception as e:
            self.logger.error(f"ERREUR PIPELINE: {e}")
            self.execution_metadata['end_time'] = datetime.now()
            self.execution_metadata['status'] = 'failed'
            self.execution_metadata['error'] = str(e)
            
            raise
    
    def _apply_smart_clustering(self, df_processed: pd.DataFrame) -> pd.DataFrame:
        """
        Application du clustering avec gestion des modèles et logging détaillé.

        LOGIQUE:
        1. En mode 'global', injecte temporairement la colonne AGE depuis les données brutes
        2. Essaie de charger un modèle existant
        3. Log toutes les métriques détaillées pour décision manuelle
        4. Recalcule SEULEMENT si force_retrain=True
        5. En mode 'global', retire la colonne AGE temporaire pour maintenir le schéma
        """
        self.logger.info("=== CLUSTERING AVEC GESTION MODÈLES ===")

        # En mode global, AGE n'est pas dans df_processed (transformé en colonnes binaires)
        # On l'injecte temporairement depuis les données brutes pour AgeClusterer
        age_was_injected = False
        if self.clustering_mode == 'global' and 'AGE' not in df_processed.columns:
            df_raw = self.results.get('raw_data')
            if df_raw is not None and 'AGE' in df_raw.columns:
                # Aligner les index et injecter AGE
                df_processed = df_processed.copy()
                df_processed['AGE'] = pd.to_numeric(df_raw['AGE'], errors='coerce').values
                age_was_injected = True
                self.logger.info("Mode global: AGE injecté temporairement pour stratification")

        clusterer = AgeClusterer()
        
        if self.force_retrain:
            # Recalcul forcé explicitement demandé
            self.logger.info("RECALCUL FORCÉ - Entraînement nouveau modèle")
            self.execution_metadata['model_action'] = 'retrained'
            self.execution_metadata['retrain_reason'] = 'Force retrain demandé'
            
            clusterer_new = AgeClusterer()
            df_result = clusterer_new.fit_transform(df_processed)
            self.metrics['clustering'] = clusterer_new.quality_metrics
            
        else:
            # Tentative de réutilisation modèle existant
            try:
                latest_model = clusterer._find_latest_model()
                if latest_model:
                    self.logger.info(f"Modèle existant trouvé: {latest_model}")
                    clusterer.load_model(latest_model)
                    
                    # Log détaillé des métriques pour décision manuelle
                    self._log_detailed_model_metrics(clusterer, df_processed)
                    
                    # Application du modèle existant
                    self.logger.info("APPLICATION MODÈLE EXISTANT")
                    self.execution_metadata['model_action'] = 'reused'
                    
                    df_result = clusterer.transform(df_processed)
                    self.metrics['clustering'] = clusterer.quality_metrics
                    
                else:
                    # Aucun modèle existant - Entraînement obligatoire
                    self.logger.info("AUCUN MODÈLE EXISTANT - Premier entraînement")
                    self.execution_metadata['model_action'] = 'retrained'
                    self.execution_metadata['retrain_reason'] = 'Aucun modèle existant'
                    
                    clusterer_new = AgeClusterer()
                    df_result = clusterer_new.fit_transform(df_processed)
                    self.metrics['clustering'] = clusterer_new.quality_metrics
                    
            except Exception as e:
                self.logger.error(f"ERREUR CHARGEMENT MODÈLE: {e}")
                self.logger.info("FALLBACK - Entraînement nouveau modèle")
                self.execution_metadata['model_action'] = 'retrained'
                self.execution_metadata['retrain_reason'] = f'Erreur chargement: {e}'
                
                clusterer_new = AgeClusterer()
                df_result = clusterer_new.fit_transform(df_processed)
                self.metrics['clustering'] = clusterer_new.quality_metrics

        # En mode global, retirer la colonne AGE temporaire pour maintenir le schéma original
        if age_was_injected and 'AGE' in df_result.columns:
            df_result = df_result.drop(columns=['AGE'])
            self.logger.info("Mode global: AGE temporaire retiré du résultat")

        return df_result
    
    def _log_detailed_model_metrics(self, clusterer: AgeClusterer, df_new: pd.DataFrame) -> None:
        """
        Log détaillé de toutes les métriques pour décision manuelle de recalcul.
        
        Args:
            clusterer: Clusterer avec modèle chargé
            df_new: Nouvelles données à traiter
        """
        self.logger.info("=== MÉTRIQUES DÉTAILLÉES DU MODÈLE ===")
        
        try:
            # Métadonnées du modèle existant
            training_meta = clusterer.training_metadata
            quality_metrics = clusterer.quality_metrics.get('global_metrics', {})
            
            self.logger.info("📊 MODÈLE EXISTANT:")
            self.logger.info(f"  Version: {clusterer.model_version}")
            self.logger.info(f"  Date entraînement: {training_meta.get('training_date', 'Inconnue')}")
            self.logger.info(f"  Échantillons entraînement: {training_meta.get('n_samples', 0):,}")
            self.logger.info(f"  Features entraînement: {training_meta.get('n_features', 0)}")
            self.logger.info(f"  Silhouette moyenne: {quality_metrics.get('avg_silhouette_score', 0):.3f}")
            self.logger.info(f"  Min/Max silhouette: {quality_metrics.get('min_silhouette_score', 0):.3f} / {quality_metrics.get('max_silhouette_score', 0):.3f}")
            self.logger.info(f"  Clusters créés: {quality_metrics.get('total_clusters', 0)}")
            self.logger.info(f"  Strates viables: {quality_metrics.get('viable_strata', 0)}")
            self.logger.info(f"  Strates cluster unique: {quality_metrics.get('single_cluster_strata', 0)}")
            
            # Profil démographique du modèle
            self.logger.info("👥 PROFIL DÉMOGRAPHIQUE MODÈLE:")
            self.logger.info(f"  Âge moyen entraînement: {training_meta.get('age_mean', 0):.1f} ± {training_meta.get('age_std', 0):.1f}")
            self.logger.info(f"  Âge min/max entraînement: {training_meta.get('age_min', 0)} / {training_meta.get('age_max', 0)}")
            
            # Analyse des nouvelles données
            current_age = df_new['AGE'].dropna()
            current_age_mean = current_age.mean()
            current_age_std = current_age.std()
            current_age_min = current_age.min()
            current_age_max = current_age.max()
            
            self.logger.info("📈 NOUVELLES DONNÉES:")
            self.logger.info(f"  Échantillons actuels: {len(df_new):,}")
            self.logger.info(f"  Features actuelles: {len([c for c in df_new.columns if c != Config.EXTRACTION['id_column']])}")
            self.logger.info(f"  Âge moyen actuel: {current_age_mean:.1f} ± {current_age_std:.1f}")
            self.logger.info(f"  Âge min/max actuel: {current_age_min} / {current_age_max}")
            
            # Comparaison et dérive
            self.logger.info("⚖️  COMPARAISON MODÈLE vs NOUVELLES DONNÉES:")
            
            # Dérive démographique
            age_mean_drift = abs(current_age_mean - training_meta.get('age_mean', current_age_mean))
            age_std_drift = abs(current_age_std - training_meta.get('age_std', current_age_std))
            volume_ratio = len(df_new) / training_meta.get('n_samples', len(df_new))
            
            self.logger.info(f"  Dérive âge moyen: {age_mean_drift:.1f} ans")
            self.logger.info(f"  Dérive écart-type âge: {age_std_drift:.1f}")
            self.logger.info(f"  Ratio volume: {volume_ratio:.2f}x ({len(df_new):,} vs {training_meta.get('n_samples', 0):,})")
            
            # Simulation application sur échantillon pour métriques
            self.logger.info("🧪 TEST APPLICATION MODÈLE:")
            try:
                # Test sur échantillon pour éviter calcul lourd
                test_sample_size = min(1000, len(df_new))
                df_test_sample = df_new.sample(n=test_sample_size, random_state=42)
                df_test_result = clusterer.transform(df_test_sample)
                
                # Métriques d'application
                total_test = len(df_test_result)
                clustered_test = (df_test_result['CLUSTER_GLOBAL'] != 'NON_CLUSTERED').sum()
                clustered_ratio = clustered_test / total_test if total_test > 0 else 0
                
                self.logger.info(f"  Échantillon testé: {total_test:,}")
                self.logger.info(f"  Clients clusterisés: {clustered_test:,} ({clustered_ratio:.1%})")
                self.logger.info(f"  Clients non-clusterisés: {total_test - clustered_test:,} ({1-clustered_ratio:.1%})")
                
                # Distribution des clusters
                if clustered_test > 0:
                    cluster_distribution = df_test_result[df_test_result['CLUSTER_GLOBAL'] != 'NON_CLUSTERED']['CLUSTER_STRATE'].value_counts()
                    self.logger.info(f"  Strates actives: {len(cluster_distribution)}")
                    for strate, count in cluster_distribution.head(5).items():
                        self.logger.info(f"    - {strate}: {count} clients")
                        
            except Exception as test_error:
                self.logger.warning(f"  Impossible de tester application: {test_error}")
            
            # Signaux d'alerte pour décision
            self.logger.info("🚨 SIGNAUX POUR DÉCISION MANUELLE:")
            
            alerts = []
            if age_mean_drift > 3:
                alerts.append(f"DÉRIVE DÉMOGRAPHIQUE - Âge moyen dévié de {age_mean_drift:.1f} ans")
            
            if volume_ratio > 2 or volume_ratio < 0.5:
                alerts.append(f"CHANGEMENT VOLUME - Ratio {volume_ratio:.1f}x par rapport à l'entraînement")
            
            if quality_metrics.get('avg_silhouette_score', 0) < 0.3:
                alerts.append(f"QUALITÉ MODÈLE FAIBLE - Silhouette {quality_metrics.get('avg_silhouette_score', 0):.3f}")
            
            single_ratio = quality_metrics.get('single_cluster_strata', 0) / max(quality_metrics.get('viable_strata', 1), 1)
            if single_ratio > 0.4:
                alerts.append(f"TROP DE CLUSTERS UNIQUES - {single_ratio:.1%} des strates")
            
            if clustered_ratio < 0.8:
                alerts.append(f"COUVERTURE CLUSTERING FAIBLE - {clustered_ratio:.1%} clusterisés")
            
            if alerts:
                self.logger.warning("  ⚠️  ALERTES DÉTECTÉES:")
                for alert in alerts:
                    self.logger.warning(f"      {alert}")
                self.logger.warning("  💡 RECOMMANDATION: Considérer un recalcul avec force_retrain=True")
            else:
                self.logger.info("  ✅ AUCUNE ALERTE - Modèle semble adapté aux nouvelles données")
                
            # Résumé pour décision
            days_since_training = (datetime.now() - datetime.fromisoformat(training_meta.get('training_date', datetime.now().isoformat()))).days
            self.logger.info("📋 RÉSUMÉ DÉCISIONNEL:")
            self.logger.info(f"  Âge du modèle: {days_since_training} jours")
            self.logger.info(f"  Qualité modèle: {'Bonne' if quality_metrics.get('avg_silhouette_score', 0) > 0.5 else 'Modérée' if quality_metrics.get('avg_silhouette_score', 0) > 0.3 else 'Faible'}")
            self.logger.info(f"  Dérive détectée: {'Oui' if age_mean_drift > 2 or volume_ratio > 1.5 or volume_ratio < 0.7 else 'Non'}")
            self.logger.info(f"  Couverture: {'Bonne' if clustered_ratio > 0.85 else 'Modérée' if clustered_ratio > 0.7 else 'Faible'}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques détaillées: {e}")
            self.logger.warning("Utilisation du modèle existant malgré l'erreur de diagnostic")
    
    def force_model_retrain(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Force le recalcul des modèles indépendamment de leur qualité.
        
        Args:
            sample_size: Taille d'échantillon pour l'entraînement
            
        Returns:
            Résultats du pipeline avec nouveau modèle
        """
        self.logger.info("=== FORCE RETRAIN DEMANDÉ ===")
        self.force_retrain = True
        self.execution_metadata['force_retrain'] = True
        
        return self.run(sample_size=sample_size)
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Retourne le statut des modèles existants.
        
        Returns:
            Dict avec informations sur les modèles
        """
        clusterer = AgeClusterer()
        
        # Recherche modèles existants
        latest_model = clusterer._find_latest_model()
        
        if not latest_model:
            return {
                'has_model': False,
                'message': 'Aucun modèle existant trouvé'
            }
        
        try:
            clusterer.load_model(latest_model)
            
            return {
                'has_model': True,
                'model_path': latest_model,
                'model_version': clusterer.model_version,
                'training_date': clusterer.training_metadata.get('training_date', 'Inconnue'),
                'n_samples': clusterer.training_metadata.get('n_samples', 0),
                'avg_silhouette': clusterer.quality_metrics.get('global_metrics', {}).get('avg_silhouette_score', 0),
                'total_clusters': clusterer.quality_metrics.get('global_metrics', {}).get('total_clusters', 0),
                'viable_strata': clusterer.quality_metrics.get('global_metrics', {}).get('viable_strata', 0)
            }
            
        except Exception as e:
            return {
                'has_model': True,
                'model_path': latest_model,
                'error': f"Erreur chargement: {e}",
                'message': 'Modèle existant mais non lisible'
            }
    
    # === MÉTHODES INCHANGÉES ===
    # (Garder toutes les autres méthodes existantes)
    
    def _extract_data(self, sample_size: Optional[int]) -> pd.DataFrame:
        """Extraction des données avec gestion mode"""
        self.logger.info("Étape 1: Extraction données")
        
        extractor = DataExtractor(use_cache=(self.mode != 'prod'))
        
        if self.mode == 'test' and sample_size:
            df = extractor.get_sample(sample_size)
        else:
            df = extractor.extract_all_data()
        
        self.logger.info(f"Données extraites: {df.shape}")
        self.results['raw_data'] = df
        self.metrics['extraction'] = {
            'n_clients': len(df),
            'n_variables': len(df.columns),
            'timestamp': datetime.now().isoformat()
        }
        
        return df
    
    def _optimize_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimisation des dimensions"""
        self.logger.info("Étape 2: Optimisation dimensions")
        
        optimizer = DimensionOptimizer(clustering_mode=self.clustering_mode)
        df_processed = optimizer.transform(df, keep_ids=True)
        
        self.logger.info(f"Dimensions optimisées: {df_processed.shape}")
        self.results['processed_data'] = df_processed
        self.metrics['optimization'] = {
            'n_features': len([c for c in df_processed.columns 
                             if c != Config.EXTRACTION['id_column']]),
            'clustering_mode': self.clustering_mode
        }
        
        return df_processed
    
    def _apply_scoring(self, df_clustered: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Application du scoring"""
        self.logger.info("Étape 4: Scoring clients")
        
        # Extraction données financières pour scoring
        extractor = DataExtractor(use_cache=True)
        df_financial = extractor.extract_financial_data()
        
        scorer = ClientScoringEngine()
        df_scored = scorer.calculate_scores(df_clustered, df_financial)
        
        self.logger.info(f"Scoring terminé: {df_scored.shape}")
        self.results['scored_data'] = df_scored
        self.metrics['scoring'] = scorer.get_scoring_metrics()
        
        return df_scored
    
    def _persist_results(self, df: pd.DataFrame) -> None:
        """
        Persistance des résultats en base (mode prod).

        Logique :
        - Ajoute colonnes DATE_EXECUTION et MOIS_ANNEE
        - Crée la table si elle n'existe pas
        - Supprime les données du mois en cours si elles existent
        - Insère les nouvelles données
        """
        if not SPARK_AVAILABLE:
            self.logger.error("Spark non disponible - impossible de persister")
            return

        self.logger.info("Persistance résultats en base")

        try:
            # Nom de la table de sortie
            output_table = f"{BaseDonnees}.segmentation_clients_scoring"

            # Ajout des colonnes temporelles
            df_to_save = df.copy()
            current_date = datetime.now()
            df_to_save['DATE_EXECUTION'] = current_date
            df_to_save['MOIS_ANNEE'] = current_date.strftime('%Y%m')  # Format YYYYMM
            df_to_save['ANNEE'] = current_date.year
            df_to_save['MOIS'] = current_date.month

            # Récupération du mois/année actuel
            mois_annee_actuel = df_to_save['MOIS_ANNEE'].iloc[0]
            self.logger.info(f"Ajout colonnes temporelles - MOIS_ANNEE: {mois_annee_actuel}")

            # Conversion en Spark DataFrame
            df_spark = spark.createDataFrame(df_to_save)

            # Vérifier si la table existe
            table_exists = spark.catalog._jcatalog.tableExists(BaseDonnees, "segmentation_clients_scoring")

            if not table_exists:
                # Création de la table
                self.logger.info(f"Création de la nouvelle table : {output_table}")
                df_spark.write.mode("overwrite").saveAsTable(output_table)
                self.logger.info(f"Table créée avec succès : {len(df_to_save):,} lignes insérées")

            else:
                # Table existante : remplacement du mois en cours
                self.logger.info(f"Table existante - Remplacement des données pour MOIS_ANNEE={mois_annee_actuel}")

                try:
                    # Lire les données existantes SANS le mois actuel
                    self.logger.info(f"Lecture des données existantes (hors mois {mois_annee_actuel})")
                    df_existing = spark.sql(f"""
                        SELECT * FROM {output_table}
                        WHERE MOIS_ANNEE != '{mois_annee_actuel}'
                    """)

                    nb_lignes_conservees = df_existing.count()
                    self.logger.info(f"Données conservées : {nb_lignes_conservees:,} lignes (autres mois)")

                    # Concaténer avec les nouvelles données
                    self.logger.info(f"Ajout des nouvelles données : {len(df_to_save):,} lignes (mois {mois_annee_actuel})")
                    df_final = df_existing.union(df_spark)

                    # Écraser toute la table avec les données combinées
                    self.logger.info("Réécriture complète de la table")
                    df_final.write.mode("overwrite").saveAsTable(output_table)
                    self.logger.info(f"Table mise à jour : {nb_lignes_conservees:,} + {len(df_to_save):,} = {nb_lignes_conservees + len(df_to_save):,} lignes")

                except Exception as e:
                    # Si la lecture échoue, on écrase complètement la table pour éviter les doublons
                    self.logger.warning(f"Impossible de lire la table existante ({e})")
                    self.logger.warning("Réécriture complète de la table pour éviter les doublons")
                    df_spark.write.mode("overwrite").saveAsTable(output_table)
                    self.logger.info(f"Table réécrite avec {len(df_to_save):,} lignes (mois {mois_annee_actuel})")

            # Vérification finale
            count_query = f"""
            SELECT COUNT(*) as total, MOIS_ANNEE
            FROM {output_table}
            GROUP BY MOIS_ANNEE
            ORDER BY MOIS_ANNEE DESC
            LIMIT 5
            """

            result = spark.sql(count_query).toPandas()
            self.logger.info("=== ÉTAT DE LA TABLE APRÈS PERSISTANCE ===")
            for _, row in result.iterrows():
                self.logger.info(f"  MOIS_ANNEE {row['MOIS_ANNEE']}: {row['total']:,} clients")

            # Métriques de persistance
            self.metrics['persistence'] = {
                'table_name': output_table,
                'mois_annee': mois_annee_actuel,
                'rows_inserted': len(df_to_save),
                'timestamp': datetime.now().isoformat(),
                'table_existed': table_exists
            }

            self.logger.info(f"✅ Persistance réussie : {len(df_to_save):,} lignes dans {output_table}")

        except Exception as e:
            self.logger.error(f"❌ ERREUR lors de la persistance : {e}")
            self.logger.error(f"Type erreur : {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback : {traceback.format_exc()}")
            raise
    
    def _save_dev_results(self, df: pd.DataFrame) -> None:
        """Sauvegarde des résultats en mode dev"""
        output_dir = Config.MODELS_DIR / "pipeline_output" / self.execution_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde données
        df.to_csv(output_dir / "scored_clients.csv", index=False)
        
        # Sauvegarde métriques
        import json
        with open(output_dir / "execution_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Sauvegarde métadonnées
        with open(output_dir / "execution_metadata.json", 'w') as f:
            json.dump(self.execution_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Résultats sauvegardés dans: {output_dir}")


# Fonctions utilitaires mises à jour
def run_advanced_pipeline(mode: str = 'dev', 
                         clustering_mode: str = 'stratifie',
                         sample_size: Optional[int] = None,
                         force_retrain: bool = False) -> Dict[str, Any]:
    """
    Exécute le pipeline avancé avec gestion intelligente des modèles.
    
    Args:
        mode: Mode d'exécution (dev, prod, test)
        clustering_mode: Mode de clustering (global, stratifie)
        sample_size: Taille échantillon (pour mode test)
        force_retrain: Force le recalcul des modèles
        
    Returns:
        Résultats du pipeline
    """
    pipeline = AdvancedBankingPipeline(
        mode=mode, 
        clustering_mode=clustering_mode,
        force_retrain=force_retrain
    )
    return pipeline.run(sample_size=sample_size)


def get_pipeline_model_status() -> Dict[str, Any]:
    """
    Retourne le statut des modèles de la pipeline.
    
    Returns:
        Dict avec informations sur les modèles existants
    """
    pipeline = AdvancedBankingPipeline()
    return pipeline.get_model_status()


# Point d'entrée direct (pour tests)
if __name__ == "__main__":
    # Exemple d'utilisation avec réutilisation de modèle
    print("=== TEST PIPELINE AVEC MODÈLE INTELLIGENT ===")
    
    # Vérifier statut des modèles
    status = get_pipeline_model_status()
    print(f"Statut modèles: {status}")
    
    # Exécution normale (réutilise modèle existant)
    results = run_advanced_pipeline(
        mode='dev',
        clustering_mode='stratifie',
        sample_size=1000,
        force_retrain=False  # Ne force pas le recalcul
    )
    
    print("Pipeline exécuté avec succès!")
    print(f"Action modèle: {results['metadata'].get('model_action', 'unknown')}")
