#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_scoring.py - Moteur de scoring client unifié

Module de scoring PNB intra-cluster avec méthodologie CA.
Calcul des scores clients, potentiels et segmentation finale.

Author: Crédit Agricole - Équipe Data Science
Date: 2025-09-18
Version: 4.2 - Intégration pipeline existante
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Import de la configuration centralisée
from .config import Config, get_logger

class ClientScoringEngine:
    """
    Moteur de scoring PNB intra-cluster avec intégration au système unifié.
    
    Fonctionnalités :
    - Pondération relative par moyennes absolues PNB cluster
    - Normalisation min-max des PNB au sein des clusters
    - Calcul potentiels avec top 25% par cluster
    - Segmentation finale (score PNB + note MIRE)
    - Métriques de qualité intégrées aux logs
    """
    
    def __init__(self):
        """
        Initialise le moteur de scoring avec la configuration centralisée.
        """
        # Configuration
        self.top_client_pct = Config.SCORING['top_client_pct']
        self.pnb_columns = Config.SCORING['pnb_columns']
        
        # Logger unifié
        self.logger = get_logger('client_scoring')
        
        # Stockage des résultats
        self.client_scores = pd.DataFrame()
        self.scoring_metrics = {}
        self.is_fitted = False
        
        self.logger.info(f"ClientScoringEngine initialisé - Top clients: {self.top_client_pct*100}%")
    
    def calculate_scores(self, df_clustered: pd.DataFrame, df_financial: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les scores clients selon méthodologie CA.
        
        Args:
            df_clustered: DataFrame avec colonnes CLUSTER_GLOBAL 
            df_financial: DataFrame avec données financières PNB
            
        Returns:
            DataFrame avec scores, potentiels et segmentation
        """
        
        self.logger.info("Début calcul scores clients intra-cluster")
        self.logger.info(f"Données clusterisées : {df_clustered.shape}")
        self.logger.info(f"Données financières : {df_financial.shape}")
        
        start_time = time.time()
        
        # Étapes de traitement
        data = self._merge_and_validate_data(df_clustered, df_financial)
        data = self._prepare_financial_data(data)
        data = self._calculate_weighted_normalized_pnb(data)
        data = self._calculate_client_score(data)
        data = self._calculate_client_potentials(data)
        data = self._calculate_final_segmentation(data)
        self._calculate_scoring_metrics(data)
        self._log_scoring_metrics()
        
        self.client_scores = data
        self.is_fitted = True
        
        total_time = time.time() - start_time
        self.logger.info(f"Scoring terminé en {total_time:.2f}s - {len(data)} clients scorés")
        
        return data
    
    def _merge_and_validate_data(self, df_clustered: pd.DataFrame, df_financial: pd.DataFrame) -> pd.DataFrame:
        """Fusionne et valide les données avec gestion d'erreurs robuste"""
        
        # Validations de base
        required_cols_clustered = [Config.EXTRACTION['id_column'], 'CLUSTER_GLOBAL']
        required_cols_financial = [Config.EXTRACTION['id_column']]
        
        missing_clustered = [col for col in required_cols_clustered if col not in df_clustered.columns]
        missing_financial = [col for col in required_cols_financial if col not in df_financial.columns]
        
        if missing_clustered:
            self.logger.error(f"Colonnes manquantes dans df_clustered: {missing_clustered}")
            raise ValueError(f"Colonnes manquantes: {missing_clustered}")
        
        if missing_financial:
            self.logger.error(f"Colonnes manquantes dans df_financial: {missing_financial}")
            raise ValueError(f"Colonnes manquantes: {missing_financial}")
        
        # Fusion
        data = df_clustered.merge(df_financial, on=Config.EXTRACTION['id_column'], how='inner')
        self.logger.info(f"Données fusionnées : {data.shape}")
        
        # Nettoyage CLUSTER_GLOBAL
        data = self._clean_cluster_column(data)
        
        # Filtrage clients clusterisés
        clustered_data = data[data['CLUSTER_GLOBAL'] != 'NON_CLUSTERED'].copy()
        
        self.logger.info(f"Clients clusterisés retenus : {len(clustered_data)}")
        self.logger.info(f"Clusters uniques : {clustered_data['CLUSTER_GLOBAL'].nunique()}")
        
        return clustered_data
    
    def _clean_cluster_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """Nettoie la colonne CLUSTER_GLOBAL de manière robuste"""
        
        self.logger.info("Nettoyage colonne CLUSTER_GLOBAL")
        
        # Conversion sécurisée
        cluster_values = []
        for val in data['CLUSTER_GLOBAL']:
            try:
                if pd.isna(val) or val in [None, '', 'nan', 'None', 'null']:
                    cluster_values.append('NON_CLUSTERED')
                elif isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) > 0:
                        cluster_values.append(str(val[0]))
                    else:
                        cluster_values.append('NON_CLUSTERED')
                else:
                    cluster_values.append(str(val))
            except Exception:
                cluster_values.append('NON_CLUSTERED')
        
        data['CLUSTER_GLOBAL'] = cluster_values
        
        # Validation finale
        try:
            test_groups = data.groupby('CLUSTER_GLOBAL').size()
            self.logger.info(f"Validation groupby réussie : {len(test_groups)} groupes")
        except Exception as e:
            self.logger.error(f"Erreur validation CLUSTER_GLOBAL: {e}")
            raise ValueError(f"Colonne CLUSTER_GLOBAL invalide: {e}")
        
        return data
    
    def _prepare_financial_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare les données financières pour le scoring"""
        
        self.logger.info("Préparation données financières")
        
        # Normalisation des noms de colonnes (case-insensitive matching)
        data_cols_lower = {col.lower(): col for col in data.columns}
        
        # Mapping des colonnes PNB disponibles
        available_pnb = {}
        for expected_col in self.pnb_columns:
            if expected_col in data.columns:
                available_pnb[expected_col] = expected_col
            elif expected_col.lower() in data_cols_lower:
                available_pnb[expected_col] = data_cols_lower[expected_col.lower()]
        
        self.logger.info(f"Colonnes PNB disponibles : {list(available_pnb.keys())}")
        
        # Normalisation des colonnes PNB
        for expected_col, actual_col in available_pnb.items():
            if actual_col != expected_col:
                data[expected_col] = data[actual_col]
            
            # Conversion numérique
            data[expected_col] = pd.to_numeric(data[expected_col], errors='coerce').fillna(0)
        
        # PNB total
        pnb_cols = list(available_pnb.keys())
        if pnb_cols:
            data['PNB_TOTAL'] = data[pnb_cols].sum(axis=1)
        else:
            self.logger.warning("Aucune colonne PNB trouvée - PNB_TOTAL = 0")
            data['PNB_TOTAL'] = 0
        
        # Autres variables financières
        financial_vars = {
            'EP_LIQUIDE': ['ep_liquide', 'EP_LIQUIDE'],
            'EP_STABLE': ['ep_stable', 'EP_STABLE'],
            'SLD_DAV': ['sld_dav', 'SLD_DAV'],
            'CD_NOTE_MIRE': ['cd_note_mire', 'CD_NOTE_MIRE']
        }
        
        for expected_var, alternatives in financial_vars.items():
            found = False
            for alt in alternatives:
                if alt in data.columns:
                    if alt != expected_var:
                        data[expected_var] = data[alt]
                    data[expected_var] = pd.to_numeric(data[expected_var], errors='coerce').fillna(0)
                    found = True
                    break
                elif alt.lower() in data_cols_lower:
                    actual_col = data_cols_lower[alt.lower()]
                    data[expected_var] = pd.to_numeric(data[actual_col], errors='coerce').fillna(0)
                    found = True
                    break
            
            if not found:
                data[expected_var] = 0
                self.logger.warning(f"Variable {expected_var} non trouvée - valeur par défaut : 0")
        
        # Calcul épargne totale et ratios
        data['TOTAL_EPARGNE'] = data['EP_LIQUIDE'] + data['EP_STABLE']
        data['EP_RATIO'] = np.where(
            data['TOTAL_EPARGNE'] > 200,
            data['SLD_DAV'].clip(lower=200) / data['TOTAL_EPARGNE'],
            0
        ).clip(0, 1)
        
        # Variables de détention par univers (déjà présentes dans les données)
        detention_vars = [
            'CD_NOTE_UNVRS_ASSU', 'CD_NOTE_UNVRS_CRED', 
            'CD_NOTE_UNVRS_EPRGN', 'CD_NOTE_UNVRS_SERV',
            'CD_NOTE_POIDS_UNVRS'
        ]
        
        for var in detention_vars:
            if var in data.columns:
                data[var] = pd.to_numeric(data[var], errors='coerce').fillna(0)
                self.logger.info(f"Variable détention {var} préparée")
            else:
                data[var] = 0
                self.logger.warning(f"Variable détention {var} non trouvée - valeur par défaut : 0")
        
        return data
    
    def _calculate_weighted_normalized_pnb(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule pondération et normalisation PNB par cluster"""
        
        self.logger.info("Calcul pondération et normalisation PNB par cluster")
        
        available_pnb_cols = [col for col in self.pnb_columns if col in data.columns]
        
        if not available_pnb_cols:
            self.logger.warning("Aucune colonne PNB pour pondération")
            return data
        
        # Moyennes absolues par cluster
        try:
            cluster_means = (data.groupby('CLUSTER_GLOBAL')[available_pnb_cols]
                           .mean().abs())
            
            # Calcul des poids relatifs par cluster
            cluster_weights = cluster_means.div(cluster_means.sum(axis=1), axis=0).fillna(0)
            
            # Merge avec données principales
            cluster_weights_reset = cluster_weights.reset_index()
            weight_cols = {col: f'WEIGHT_{col}' for col in available_pnb_cols}
            cluster_weights_reset = cluster_weights_reset.rename(columns=weight_cols)
            
            data = data.merge(cluster_weights_reset, on='CLUSTER_GLOBAL', how='left')
            
        except Exception as e:
            self.logger.error(f"Erreur calcul pondération: {e}")
            # Poids égaux par défaut
            for col in available_pnb_cols:
                data[f'WEIGHT_{col}'] = 1.0 / len(available_pnb_cols)
        
        # Normalisation min-max par cluster
        for col in available_pnb_cols:
            try:
                col_grouped = data.groupby('CLUSTER_GLOBAL')[col]
                col_min = col_grouped.transform('min')
                col_max = col_grouped.transform('max')
                
                # Normalisation avec protection division par zéro
                data[f'{col}_NORM'] = np.where(
                    col_max - col_min > 1e-6,
                    (data[col] - col_min) / (col_max - col_min),
                    0
                )
                
                # Application pondération
                weight_col = f'WEIGHT_{col}'
                if weight_col in data.columns:
                    data[f'{col}_WEIGHTED'] = data[f'{col}_NORM'] * data[weight_col]
                else:
                    data[f'{col}_WEIGHTED'] = data[f'{col}_NORM'] / len(available_pnb_cols)
                    
            except Exception as e:
                self.logger.warning(f"Erreur normalisation {col}: {e}")
                data[f'{col}_NORM'] = 0
                data[f'{col}_WEIGHTED'] = 0
        
        return data
    
    def _calculate_client_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule le score client pondéré"""
        
        available_pnb_cols = [col for col in self.pnb_columns if col in data.columns]
        weighted_cols = [f'{col}_WEIGHTED' for col in available_pnb_cols 
                        if f'{col}_WEIGHTED' in data.columns]
        
        if weighted_cols:
            data['CLIENT_SCORE'] = data[weighted_cols].sum(axis=1)
            self.logger.info(f"Score client calculé avec {len(weighted_cols)} composantes")
        else:
            data['CLIENT_SCORE'] = 0
            self.logger.warning("Aucune composante pondérée pour le score client")
        
        return data
    
    def _calculate_client_potentials(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule les potentiels par cluster selon méthodologie CA"""

        self.logger.info("Calcul potentiels clients (top 25% par cluster)")

        available_pnb_cols = [col for col in self.pnb_columns if col in data.columns]

        try:
            # Identification top 25% par cluster
            def get_top_clients_indices(group):
                n_top = max(1, int(len(group) * self.top_client_pct))
                top_clients = group.nlargest(n_top, 'CLIENT_SCORE')
                return top_clients.index.tolist()  # Convertir en liste ici

            # Récupération et aplatissement des indices
            top_indices_by_cluster = (data.groupby('CLUSTER_GLOBAL', group_keys=False)
                                    .apply(get_top_clients_indices))

            # Aplatir la liste des listes d'indices
            top_client_indices = []
            for indices_list in top_indices_by_cluster:
                top_client_indices.extend(indices_list)

            # Calcul médianes PNB des top clients par cluster
            if available_pnb_cols:
                median_pnb = (data.loc[top_client_indices]
                            .groupby('CLUSTER_GLOBAL')[available_pnb_cols]
                            .median())

                # Renommage colonnes médianes
                median_cols = {col: f'{col}_MEDIAN' for col in available_pnb_cols}
                median_pnb = median_pnb.rename(columns=median_cols).reset_index()

                # Merge avec données principales
                data = data.merge(median_pnb, on='CLUSTER_GLOBAL', how='left')

            # Médiane ratio épargne pour top clients
            median_ep_ratio = (data.loc[top_client_indices]
                             .groupby('CLUSTER_GLOBAL')['EP_RATIO']
                             .median().reset_index()
                             .rename(columns={'EP_RATIO': 'EP_RATIO_MEDIAN'}))

            data = data.merge(median_ep_ratio, on='CLUSTER_GLOBAL', how='left')

            # Calcul potentiels par univers
            potential_cols = []

            for col in available_pnb_cols:
                median_col = f'{col}_MEDIAN'
                potential_col = f'POTENTIAL_{col}'

                if median_col in data.columns:
                    if col == 'PNB_COLL':
                        # Collecte : potentiel basé sur sous-captation épargne
                        is_under_captated = data['EP_RATIO'] > data['EP_RATIO_MEDIAN'].fillna(0)
                        dynamic_gap = np.where(
                            is_under_captated,
                            (data['EP_RATIO'] - data['EP_RATIO_MEDIAN'].fillna(0)) / 
                            (1 - data['EP_RATIO_MEDIAN'].fillna(0) + 1e-6),
                            0
                        )
                        data[potential_col] = dynamic_gap * data[median_col].fillna(0)

                    elif col == 'PNB_CRED': 
                        # Crédit : potentiel seulement si pas de détention crédit (CD_NOTE_UNVRS_CRED == 0)
                        has_no_credit = data['CD_NOTE_UNVRS_CRED'].fillna(0) == 0
                        base_potential = (data[median_col].fillna(0) - 
                                        data[col].clip(lower=0)).clip(lower=0)
                        data[potential_col] = np.where(has_no_credit, base_potential, 0)

                    else:
                        # Autres univers : écart à la médiane
                        data[potential_col] = (data[median_col].fillna(0) - 
                                             data[col].clip(lower=0)).clip(lower=0)

                    potential_cols.append(potential_col)
                else:
                    data[potential_col] = 0

            # Potentiel total
            if potential_cols:
                data['TOTAL_POTENTIAL'] = data[potential_cols].sum(axis=1)
            else:
                data['TOTAL_POTENTIAL'] = 0

            n_top_clients = len(top_client_indices)
            self.logger.info(f"Potentiels calculés basés sur {n_top_clients} clients top 25%")

        except Exception as e:
            self.logger.error(f"Erreur calcul potentiels: {e}")
            data['TOTAL_POTENTIAL'] = 0
            for col in available_pnb_cols:
                data[f'POTENTIAL_{col}'] = 0

        return data
    
    def _calculate_final_segmentation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule la segmentation finale combinée"""
        
        self.logger.info("Calcul segmentation finale")
        
        try:
            # Statistiques score par cluster pour segmentation PNB
            score_stats = (data.groupby('CLUSTER_GLOBAL')['CLIENT_SCORE']
                         .agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
                         .reset_index())
            
            score_stats.columns = ['CLUSTER_GLOBAL', 'SCORE_MEDIAN', 'SCORE_Q25', 'SCORE_Q75']
            data = data.merge(score_stats, on='CLUSTER_GLOBAL', how='left')
            
            # Segment PNB (1-4 selon position dans cluster)
            def calculate_pnb_segment(row):
                score = row['CLIENT_SCORE']
                if pd.isna(score) or pd.isna(row['SCORE_Q75']):
                    return 2
                
                if score >= row['SCORE_Q75']:
                    return 4
                elif score >= row['SCORE_MEDIAN']:
                    return 3
                elif score >= row['SCORE_Q25']:
                    return 2
                else:
                    return 1
            
            data['SEGMENT_PNB'] = data.apply(calculate_pnb_segment, axis=1)
            
            # Segment MIRE (selon note globale)
            def calculate_mire_segment(note):
                if pd.isna(note):
                    return 2
                if note >= 75:
                    return 4
                elif note >= 50:
                    return 3
                elif note >= 25:
                    return 2
                else:
                    return 1
            
            data['SEGMENT_MIRE'] = data['CD_NOTE_MIRE'].apply(calculate_mire_segment)
            
            # Segmentation finale combinée
            data['SEGMENT_FINAL'] = data['SEGMENT_PNB'] + data['SEGMENT_MIRE']
            
            # Libellé final
            def get_final_label(combined_score):
                if pd.isna(combined_score):
                    return 'A construire'
                if combined_score == 8:
                    return 'A entretenir'
                elif combined_score >= 6:
                    return 'A developper'
                elif combined_score >= 4:
                    return 'A stimuler'
                else:
                    return 'A construire'
            
            data['SEGMENTATION_FINALE'] = data['SEGMENT_FINAL'].apply(get_final_label)
            
        except Exception as e:
            self.logger.error(f"Erreur segmentation finale: {e}")
            data['SEGMENT_PNB'] = 2
            data['SEGMENT_MIRE'] = 2
            data['SEGMENT_FINAL'] = 4
            data['SEGMENTATION_FINALE'] = 'A stimuler'
        
        return data
    
    def _calculate_scoring_metrics(self, data: pd.DataFrame):
        """Calcule les métriques de qualité du scoring"""
        
        if data.empty:
            self.scoring_metrics = {'error': 'Aucune donnée pour les métriques'}
            return
        
        # Métriques globales
        total_clients = len(data)
        total_potential = data['TOTAL_POTENTIAL'].sum()
        avg_score = data['CLIENT_SCORE'].mean()
        
        # Distribution des segments
        segment_distribution = data['SEGMENTATION_FINALE'].value_counts().to_dict()
        
        # Métriques par cluster
        cluster_metrics = {}
        for cluster in data['CLUSTER_GLOBAL'].unique():
            cluster_data = data[data['CLUSTER_GLOBAL'] == cluster]
            cluster_metrics[cluster] = {
                'n_clients': len(cluster_data),
                'avg_score': cluster_data['CLIENT_SCORE'].mean(),
                'total_potential': cluster_data['TOTAL_POTENTIAL'].sum(),
                'avg_potential': cluster_data['TOTAL_POTENTIAL'].mean(),
                'dominant_segment': cluster_data['SEGMENTATION_FINALE'].mode().iloc[0] if not cluster_data.empty else 'Unknown'
            }
        
        # Top clusters par potentiel
        top_clusters = sorted(cluster_metrics.items(), 
                            key=lambda x: x[1]['avg_potential'], 
                            reverse=True)[:5]
        
        # Métriques de qualité
        non_zero_scores = (data['CLIENT_SCORE'] > 0).sum()
        score_coverage = non_zero_scores / total_clients if total_clients > 0 else 0
        
        clients_with_potential = (data['TOTAL_POTENTIAL'] > 0).sum()
        potential_coverage = clients_with_potential / total_clients if total_clients > 0 else 0
        
        self.scoring_metrics = {
            'timestamp': datetime.now().isoformat(),
            'global_metrics': {
                'total_clients': total_clients,
                'clusters_count': data['CLUSTER_GLOBAL'].nunique(),
                'avg_client_score': avg_score,
                'total_potential_value': total_potential,
                'avg_potential_per_client': total_potential / total_clients if total_clients > 0 else 0,
                'score_coverage': score_coverage,
                'potential_coverage': potential_coverage
            },
            'segment_distribution': segment_distribution,
            'cluster_metrics': cluster_metrics,
            'top_clusters_by_potential': [{'cluster': k, **v} for k, v in top_clusters],
            'quality_indicators': {
                'clients_scored': non_zero_scores,
                'clients_with_potential': clients_with_potential,
                'segments_balanced': len(segment_distribution) >= 3,
                'clusters_viable': len([c for c in cluster_metrics.values() if c['n_clients'] >= 50])
            }
        }
    
    def _log_scoring_metrics(self):
        """Log détaillé des métriques de scoring"""
        
        if 'error' in self.scoring_metrics:
            self.logger.error(f"Métriques scoring : {self.scoring_metrics['error']}")
            return
        
        gm = self.scoring_metrics['global_metrics']
        
        self.logger.info("=== MÉTRIQUES DE QUALITÉ SCORING ===")
        self.logger.info(f"Clients scorés : {gm['total_clients']:,}")
        self.logger.info(f"Clusters analysés : {gm['clusters_count']}")
        self.logger.info(f"Score client moyen : {gm['avg_client_score']:.3f}")
        self.logger.info(f"Potentiel total : {gm['total_potential_value']:,.0f}€")
        self.logger.info(f"Potentiel moyen/client : {gm['avg_potential_per_client']:,.0f}€")
        self.logger.info(f"Couverture scoring : {gm['score_coverage']:.1%}")
        self.logger.info(f"Couverture potentiel : {gm['potential_coverage']:.1%}")
        
        # Distribution segments
        self.logger.info("Distribution segments :")
        for segment, count in self.scoring_metrics['segment_distribution'].items():
            pct = count / gm['total_clients'] * 100
            self.logger.info(f"  - {segment}: {count:,} clients ({pct:.1f}%)")
        
        # Top clusters
        self.logger.info("Top 3 clusters par potentiel moyen :")
        for cluster_info in self.scoring_metrics['top_clusters_by_potential'][:3]:
            self.logger.info(f"  - {cluster_info['cluster']}: "
                           f"{cluster_info['avg_potential']:,.0f}€ "
                           f"({cluster_info['n_clients']} clients)")
    
    def get_scoring_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de scoring complètes"""
        if not self.is_fitted:
            self.logger.warning("Modèle non entraîné - métriques indisponibles")
            return {}
        
        return self.scoring_metrics
    
    def export_results(self, filepath: str = None) -> str:
        """Exporte les résultats de scoring"""
        if not self.is_fitted:
            self.logger.error("Aucun résultat à exporter")
            raise ValueError("Aucun résultat de scoring disponible")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Config.MODELS_DIR / f"client_scoring_results_{timestamp}.csv"
        
        # Colonnes essentielles pour export
        essential_cols = [
            Config.EXTRACTION['id_column'], 'CLUSTER_GLOBAL', 'CLIENT_SCORE', 
            'TOTAL_POTENTIAL', 'SEGMENTATION_FINALE', 'SEGMENT_PNB', 'SEGMENT_MIRE'
        ]
        
        available_cols = [col for col in essential_cols if col in self.client_scores.columns]
        
        if available_cols:
            export_data = self.client_scores[available_cols].copy()
            export_data.to_csv(filepath, index=False)
            self.logger.info(f"Résultats exportés : {filepath}")
        else:
            self.logger.error("Aucune colonne essentielle trouvée pour export")
            raise ValueError("Données d'export introuvables")
        
        return str(filepath)
      
    def export_completion_csv(self, filepath: str = None) -> str:
        """
        Exporte les données client au format simple pour interface web.

        Structure exportée :
        - ID client
        - segmentNumber (0-4)
        - pnbPercentage (0-100%)
        - Scores PNB par univers (pnbCredit, pnbBAQ, pnbAssurance, pnbEpargne)
        - Scores Détention par univers (detentionCredit, detentionBAQ, detentionAssurance, detentionEpargne) 
        - notesMire (CD_NOTE_MIRE direct)

        Args:
            filepath: Chemin du fichier CSV à créer

        Returns:
            Chemin du fichier créé
        """
        if not self.is_fitted:
            self.logger.error("Aucun résultat à exporter")
            raise ValueError("Aucun résultat de scoring disponible")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Config.MODELS_DIR / f"client_data_simple_{timestamp}.csv"

        # Copie des données
        export_data = self.client_scores.copy()

        # 1. Mapping segment final vers numéro (0-4)
        segment_mapping = {
            'A construire': 0,
            'A stimuler': 1, 
            'A developper': 2,
            'A entretenir': 3
        }

        export_data['segmentNumber'] = export_data['SEGMENTATION_FINALE'].map(segment_mapping).fillna(1)

        # 2. Calcul scores PNB (0-100) basés sur les colonnes PNB normalisées par cluster
        univers_pnb_mapping = {
            'pnbCredit': ['PNB_CRED', 'PNB_CRED_NORM'],
            'pnbBAQ': ['PNB_SERV', 'PNB_SERV_NORM'], 
            'pnbAssurance': ['PNB_ASSU', 'PNB_ASSU_NORM'],
            'pnbEpargne': ['PNB_COLL', 'PNB_COLL_NORM']
        }

        # Pour chaque univers PNB, utiliser la colonne normalisée si disponible, sinon calculer
        for univers, cols in univers_pnb_mapping.items():
            # Essayer d'abord la colonne normalisée
            norm_col = cols[1] if len(cols) > 1 else None
            raw_col = cols[0]

            if norm_col in export_data.columns:
                # Utiliser la valeur normalisée (0-1) et convertir en 0-100
                export_data[univers] = (export_data[norm_col] * 100).round().astype(int)
            elif raw_col in export_data.columns:
                # Normaliser manuellement par cluster et convertir en 0-100
                try:
                    grouped = export_data.groupby('CLUSTER_GLOBAL')[raw_col]
                    col_min = grouped.transform('min')
                    col_max = grouped.transform('max')
                    normalized = np.where(
                        col_max - col_min > 1e-6,
                        (export_data[raw_col] - col_min) / (col_max - col_min),
                        0.5  # Valeur par défaut si pas de variation
                    )
                    export_data[univers] = (normalized * 100).round().astype(int)
                except:
                    # Valeur par défaut
                    export_data[univers] = 50
            else:
                # Valeur par défaut si colonne inexistante
                export_data[univers] = 50

        # 3. Calcul scores Détention (0-100) basés sur CD_NOTE_UNVRS_* normalisés par max global
        univers_detention_mapping = {
            'detentionCredit': 'CD_NOTE_UNVRS_CRED',
            'detentionBAQ': 'CD_NOTE_UNVRS_SERV',
            'detentionAssurance': 'CD_NOTE_UNVRS_ASSU', 
            'detentionEpargne': 'CD_NOTE_UNVRS_EPRGN'
        }

        for detention_score, source_col in univers_detention_mapping.items():
            if source_col in export_data.columns:
                # Normaliser par le maximum global (pas par cluster)
                col_max = export_data[source_col].max()
                if col_max > 0:
                    normalized = (export_data[source_col].fillna(0) / col_max)
                    export_data[detention_score] = (normalized * 100).round().astype(int)
                else:
                    export_data[detention_score] = 0
            else:
                # Colonnes pas encore récupérées du module extractor
                export_data[detention_score] = 50

        # 4. Calcul pourcentage PNB (basé sur CLIENT_SCORE normalisé par cluster)
        try:
            # Normaliser CLIENT_SCORE par cluster (0-1)
            grouped_score = export_data.groupby('CLUSTER_GLOBAL')['CLIENT_SCORE']
            score_min = grouped_score.transform('min')
            score_max = grouped_score.transform('max')

            pnb_normalized = np.where(
                score_max - score_min > 1e-6,
                (export_data['CLIENT_SCORE'] - score_min) / (score_max - score_min),
                0.5
            )
            export_data['pnbPercentage'] = (pnb_normalized * 100).round().astype(int)
        except:
            # Valeur par défaut
            export_data['pnbPercentage'] = 75

        # 5. Ajout CD_NOTE_MIRE direct
        if 'CD_NOTE_MIRE' in export_data.columns:
            export_data['notesMire'] = export_data['CD_NOTE_MIRE'].fillna(0).astype(int)
        else:
            export_data['notesMire'] = 50

        # 6. Préparation du CSV final
        final_data = pd.DataFrame({
            'clientId': export_data[Config.EXTRACTION['id_column']],
            'segmentNumber': export_data['segmentNumber'].astype(int),
            'pnbPercentage': export_data['pnbPercentage'].astype(int),
            'pnbCredit': export_data['pnbCredit'].astype(int),
            'pnbBAQ': export_data['pnbBAQ'].astype(int), 
            'pnbAssurance': export_data['pnbAssurance'].astype(int),
            'pnbEpargne': export_data['pnbEpargne'].astype(int),
            'detentionCredit': export_data['detentionCredit'].astype(int),
            'detentionBAQ': export_data['detentionBAQ'].astype(int),
            'detentionAssurance': export_data['detentionAssurance'].astype(int),
            'detentionEpargne': export_data['detentionEpargne'].astype(int),
            'notesMire': export_data['notesMire'].astype(int)
        })

        # 7. Export CSV
        final_data.to_excel(filepath, index=False)

        self.logger.info(f"Export CSV simple créé : {filepath}")
        self.logger.info(f"Données exportées : {len(final_data)} clients")
        self.logger.info("Colonnes : clientId, segmentNumber, pnbPercentage, scores PNB (4), scores détention (4), notesMire")

        return str(filepath)    
    
# Fonctions utilitaires
def quick_client_scoring(df_clustered: pd.DataFrame, df_financial: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fonction utilitaire pour scoring rapide"""
    scorer = ClientScoringEngine()
    df_result = scorer.calculate_scores(df_clustered, df_financial)
    metrics = scorer.get_scoring_metrics()
    return df_result, metrics