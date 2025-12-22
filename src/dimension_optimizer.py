#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dimension_optimizer.py - Optimiseur de dimensions pour clustering bancaire

Support des modes global et stratifié.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
import warnings
warnings.filterwarnings('ignore')

from .config import Config, get_logger


class DimensionOptimizer:
    """
    Optimiseur de dimensions avec modes global et stratifié.

    Modes disponibles :
    - "global" : 11 dimensions avec tranches d'âge encodées
    - "stratifie" : 8 dimensions + AGE numérique pour stratification
    """
    
    def __init__(self, clustering_mode: str = "global"):
        """
        Initialise l'optimiseur avec le mode de clustering choisi.
        
        Args:
            clustering_mode: "global" pour clustering classique ou "stratifie" pour clustering par strates d'âge
        """
        
        if clustering_mode not in ["global", "stratifie"]:
            raise ValueError(f"Mode de clustering invalide : {clustering_mode}. Utilisez 'global' ou 'stratifie'")
        
        self.clustering_mode = clustering_mode
        
        # Logger unifié
        self.logger = get_logger('dimension_optimizer')
        
        # Configuration selon le mode
        if clustering_mode == "global":
            self.age_bins = [
                (18, 30, 'Jeunes'),
                (30, 50, 'Actifs'),
                (50, 65, 'Seniors'),
                (65, 100, 'Retraites')
            ]
            self.total_dimensions = 11
            self.encode_age_ranges = True

        else:  # Mode stratifié
            self.age_bins = None
            self.total_dimensions = 8
            self.encode_age_ranges = False
        
        # Configuration CSP commune
        self.csp_groups = {
            'CSP_Plus': ['2', '3'],
            'CSP_Moyen': ['1', '4', '5', '6'],
            'CSP_Moins': ['7', '8', '9', 'NC']
        }
        
        self.logger.info(f"DimensionOptimizer initialisé - Mode : {clustering_mode}")
    
    def transform(self, df: pd.DataFrame, keep_ids: bool = True) -> pd.DataFrame:
        """
        Transforme le DataFrame selon le mode de clustering choisi.
        
        Args:
            df: DataFrame avec les variables socio-démographiques
            keep_ids: Si True, conserve les colonnes ID
            
        Returns:
            DataFrame transformé selon le mode choisi
        """
        
        self.logger.info(f"Début transformation dimensionnelle - Mode {self.clustering_mode}")
        self.logger.info(f"Shape initiale : {df.shape}")
        
        if df.empty:
            self.logger.error("DataFrame vide en entrée")
            return pd.DataFrame()
        
        # Copie de travail
        df_work = df.copy()
        
        # Sauvegarde des IDs
        id_columns = []
        if keep_ids and Config.EXTRACTION['id_column'] in df_work.columns:
            id_columns.append(df_work[[Config.EXTRACTION['id_column']]])
        
        # Liste des colonnes encodées
        encoded_columns = []
        
        # Transformations selon l'ordre défini
        transformations = [
            ('situation_composite', self._encode_situation_composite),
            ('age', self._encode_age_variable),
            ('revenu', self._encode_revenu_ordinal),
            ('enfants', self._encode_enfants_variable),
            ('logement', self._encode_logement_variable)
        ]
        
        for name, transform_func in transformations:
            try:
                result = transform_func(df_work)
                if not result.empty:
                    encoded_columns.append(result)
                    self.logger.info(f"Transformation {name} : {result.shape[1]} colonnes")
                else:
                    self.logger.warning(f"Transformation {name} : résultat vide")
            except Exception as e:
                self.logger.error(f"Erreur transformation {name} : {e}")
        
        # Concaténation finale
        if not encoded_columns:
            self.logger.error("Aucune transformation réussie")
            return pd.DataFrame()
        
        if id_columns:
            df_final = pd.concat(id_columns + encoded_columns, axis=1)
        else:
            df_final = pd.concat(encoded_columns, axis=1)
        
        # Vérification finale
        feature_cols = [c for c in df_final.columns if c != Config.EXTRACTION['id_column']]
        actual_dims = len(feature_cols)
        
        self.logger.info(f"Transformation terminée : {df_final.shape}")
        self.logger.info(f"Dimensions créées : {actual_dims}")
        self.logger.info(f"Colonnes finales : {list(feature_cols)}")
        
        return df_final
    
    def _encode_situation_composite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode la situation composite (couple/célibataire)"""
        encoded = pd.DataFrame(index=df.index)
        is_couple = pd.Series(False, index=df.index)
        
        # Condition 1 : 2 majeurs ou plus
        if 'nb_majeur' in df.columns:
            nb_majeur_num = pd.to_numeric(df['nb_majeur'], errors='coerce').fillna(1)
            is_couple |= (nb_majeur_num >= 2)
        
        # Condition 2 : Situation familiale "en couple"
        if 'CDSIFA' in df.columns:
            situation_norm = df['CDSIFA'].astype(str).str.lower().fillna('')
            couple_keywords = ['marié', 'marie', 'concubin', 'pacsé', 'pacse']
            for keyword in couple_keywords:
                is_couple |= situation_norm.str.contains(keyword, na=False)
        
        encoded['Situation_Couple'] = is_couple.astype(int)
        encoded['Situation_Celibataire'] = (~is_couple).astype(int)
        
        return encoded
    
    def _encode_age_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode l'âge selon le mode choisi"""
        if 'AGE' not in df.columns:
            self.logger.warning("Colonne AGE manquante")
            return pd.DataFrame(index=df.index)

        age_numeric = pd.to_numeric(df['AGE'], errors='coerce')

        if self.clustering_mode == "global":
            # Tranches d'âge binaires (sans AGE numérique pour éviter les problèmes de schéma)
            encoded = pd.DataFrame(index=df.index)
            for _, _, label in self.age_bins:
                encoded[f'Age_{label}'] = 0

            encoded.loc[(age_numeric >= 18) & (age_numeric < 30), 'Age_Jeunes'] = 1
            encoded.loc[(age_numeric >= 30) & (age_numeric < 50), 'Age_Actifs'] = 1
            encoded.loc[(age_numeric >= 50) & (age_numeric < 65), 'Age_Seniors'] = 1
            encoded.loc[age_numeric >= 65, 'Age_Retraites'] = 1

            return encoded

        else:
            # Mode stratifié : âge numérique
            return pd.DataFrame({'AGE': age_numeric}, index=df.index)
    
    def _encode_revenu_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode les revenus en variable ordinale unique"""
        if 'FRONTALIER' not in df.columns or 'CDPCSP' not in df.columns:
            self.logger.warning("Colonnes FRONTALIER ou CDPCSP manquantes pour REVENU")
            return pd.DataFrame({'REVENU_Score': [0] * len(df)}, index=df.index)
        
        # Mapping CSP
        csp_to_level = {}
        for code in self.csp_groups['CSP_Plus']:
            csp_to_level[str(code)] = 2
        for code in self.csp_groups['CSP_Moyen']:
            csp_to_level[str(code)] = 1
        for code in self.csp_groups['CSP_Moins']:
            csp_to_level[str(code)] = 0
        
        csp_level = df['CDPCSP'].astype(str).map(csp_to_level).fillna(0)
        frontalier = pd.to_numeric(df['FRONTALIER'], errors='coerce').fillna(0)
        
        # Score final : Frontalier = 3, sinon niveau CSP
        revenu_score = np.where(frontalier == 1, 3, csp_level)
        
        return pd.DataFrame({'REVENU_Score': revenu_score}, index=df.index)
    
    def _encode_enfants_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode le nombre d'enfants en binaire"""
        if 'nb_enfant' not in df.columns:
            self.logger.warning("Colonne nb_enfant manquante")
            encoded = pd.DataFrame(index=df.index)
            encoded['Enfants_Avec'] = 0
            encoded['Enfants_Sans'] = 1
            return encoded
        
        enfant_numeric = pd.to_numeric(df['nb_enfant'], errors='coerce').fillna(0)
        has_children = enfant_numeric > 0
        
        encoded = pd.DataFrame(index=df.index)
        encoded['Enfants_Avec'] = has_children.astype(int)
        encoded['Enfants_Sans'] = (~has_children).astype(int)
        
        return encoded
    
    def _encode_logement_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode le type de logement en binaire"""
        if 'CD_OCC_LGT' not in df.columns:
            self.logger.warning("Colonne CD_OCC_LGT manquante")
            encoded = pd.DataFrame(index=df.index)
            encoded['Logement_Proprietaire'] = 0
            encoded['Logement_NonProprietaire'] = 1
            return encoded
        
        logement_str = df['CD_OCC_LGT'].astype(str)
        is_proprietaire = logement_str == '1'
        
        encoded = pd.DataFrame(index=df.index)
        encoded['Logement_Proprietaire'] = is_proprietaire.astype(int)
        encoded['Logement_NonProprietaire'] = (~is_proprietaire).astype(int)
        
        return encoded


class DataPipeline:
    """Pipeline complète : extraction + optimisation dimensionnelle"""
    
    def __init__(self, clustering_mode: str = "global", use_cache: bool = True):
        """
        Initialise la pipeline complète
        
        Args:
            clustering_mode: Mode pour l'optimiseur de dimensions
            use_cache: Utilisation du cache pour l'extracteur
        """
        
        self.clustering_mode = clustering_mode
        self.use_cache = use_cache
        
        # Optimiseur immédiat
        self.optimizer = DimensionOptimizer(clustering_mode=clustering_mode)
        
        # Logger unifié
        
        self.logger.info(f"DataPipeline initialisée - Mode clustering : {clustering_mode}")
    
    def run_complete_pipeline(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Exécute la pipeline complète : extraction + transformation
        
        Args:
            sample_size: Taille d'échantillon (None pour toutes les données)
            
        Returns:
            DataFrame prêt pour le clustering
        """
        
        self.logger.info("Démarrage pipeline complète")
        
        # Import tardif pour éviter la circularité
        from .data_extractor import DataExtractor
        
        # Étape 1 : Extraction
        extractor = DataExtractor(use_cache=self.use_cache)
        
        if sample_size:
            df_raw = extractor.get_sample(sample_size)
            self.logger.info(f"Échantillon extrait : {len(df_raw)} lignes")
        else:
            df_raw = extractor.extract_all_data()
            self.logger.info(f"Données complètes extraites : {len(df_raw)} lignes")
        
        if df_raw.empty:
            self.logger.error("Aucune donnée extraite - Pipeline interrompue")
            return pd.DataFrame()
        
        # Étape 2 : Optimisation dimensionnelle
        df_processed = self.optimizer.transform(df_raw)
        
        if df_processed.empty:
            self.logger.error("Échec de l'optimisation - Pipeline interrompue")
            return pd.DataFrame()
        
        self.logger.info("Pipeline complète terminée avec succès")
        return df_processed


# Fonctions utilitaires
def run_pipeline(clustering_mode: str = "global", sample_size: Optional[int] = None) -> pd.DataFrame:
    """Fonction utilitaire pour exécuter rapidement la pipeline complète"""
    pipeline = DataPipeline(clustering_mode=clustering_mode)
    return pipeline.run_complete_pipeline(sample_size=sample_size)
