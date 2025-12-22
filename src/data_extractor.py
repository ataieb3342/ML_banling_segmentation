#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_extractor.py - Module d'extraction des données

Extraction unifiée des données organisationnelles, financières et socles.
Gestion intelligente du cache basée sur mois/année.
"""

import os
import sys
import pickle
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from .config import Config, get_logger


def get_current_period() -> str:
    """Retourne la période courante au format YYYYMM."""
    return datetime.now().strftime('%Y%m')

# Configuration Spark CA
sys.path.append('/data/FCP10/Donnees/MC/explo/FCP10_MLCDP_WAX647/code/Parametrage/')
try:
    from SessionSpark import spark, BaseDonnees
    SPARK_AVAILABLE = True
except ImportError:
    spark = None
    BaseDonnees = Config.DATABASE
    SPARK_AVAILABLE = False


class CacheManager:
    """Gestionnaire de cache avec validation mois/année."""

    def __init__(self, logger):
        self.logger = logger

    def save_cache(self, cache_path: str, data: pd.DataFrame, period: str) -> bool:
        """
        Sauvegarde les données avec métadonnées de période.

        Args:
            cache_path: Chemin du fichier cache
            data: DataFrame à sauvegarder
            period: Période au format YYYYMM

        Returns:
            True si succès, False sinon
        """
        try:
            cache_data = {
                'period': period,
                'created_at': datetime.now().isoformat(),
                'n_rows': len(data),
                'data': data
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            self.logger.info(f"Cache sauvegardé: {cache_path} (période {period}, {len(data):,} lignes)")
            return True
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde cache: {e}")
            return False

    def load_cache(self, cache_path: str, expected_period: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Charge le cache si la période correspond.

        Args:
            cache_path: Chemin du fichier cache
            expected_period: Période attendue (YYYYMM)

        Returns:
            (DataFrame ou None, message de statut)
        """
        try:
            if not os.path.exists(cache_path):
                return None, "CACHE_NOT_FOUND"

            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Vérifier si c'est l'ancien format (sans métadonnées)
            if isinstance(cache_data, pd.DataFrame):
                self.logger.warning("Cache ancien format détecté (sans période), extraction nécessaire")
                return None, "CACHE_OLD_FORMAT"

            # Vérifier la période
            cache_period = cache_data.get('period', '')
            if cache_period != expected_period:
                self.logger.info(f"Cache périmé: période {cache_period} != période courante {expected_period}")
                return None, f"CACHE_EXPIRED:{cache_period}"

            # Cache valide
            data = cache_data['data']
            created_at = cache_data.get('created_at', 'inconnue')
            self.logger.info(f"Cache valide chargé: période {cache_period}, créé le {created_at}, {len(data):,} lignes")
            return data, "CACHE_VALID"

        except Exception as e:
            self.logger.error(f"Erreur lecture cache: {e}")
            return None, f"CACHE_ERROR:{e}"

    def invalidate_cache(self, cache_path: str) -> bool:
        """Supprime le fichier cache."""
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                self.logger.info(f"Cache invalidé: {cache_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur invalidation cache: {e}")
            return False


class DataExtractor:
    """Extracteur unifié pour toutes les données avec cache intelligent mois/année."""

    # Requêtes SQL
    SQL_QUERIES = {
        'core': """
        WITH 
        LGT AS (
            SELECT ID_DWR_CLI_CIAL, CD_OCC_LGT
            FROM (
                SELECT 
                    ID_DWR_CLI_CIAL,
                    TRIM(CD_OCC_LGT) as CD_OCC_LGT,
                    ROW_NUMBER() OVER (
                        PARTITION BY ID_DWR_CLI_CIAL 
                        ORDER BY CASE
                            WHEN TRIM(CD_OCC_LGT) = '1' THEN 1
                            WHEN TRIM(CD_OCC_LGT) = '2' THEN 2
                            WHEN TRIM(CD_OCC_LGT) = '6' THEN 3
                            ELSE 11
                        END
                    ) AS row_num
                FROM {db}.connaissance_client
            ) tmp
            WHERE row_num = 1
        ),
        SIFA AS (
            SELECT ID_DWR_CLI_CIAL, CDSIFA
            FROM (
                SELECT 
                    ID_DWR_CLI_CIAL,
                    CDSIFA,
                    ROW_NUMBER() OVER (
                        PARTITION BY ID_DWR_CLI_CIAL 
                        ORDER BY CASE
                            WHEN CDSIFA = "Célibataire" THEN 1
                            WHEN CDSIFA = "Marié" THEN 2
                            WHEN CDSIFA = "Concubin" THEN 3
                            ELSE 7
                        END
                    ) AS row_num
                FROM {db}.connaissance_client
            ) tmp
            WHERE row_num = 1
        ),
        CSP AS (
            SELECT ID_DWR_CLI_CIAL, substring(lpad(CAST(CDPCSP AS STRING), 2, '0'), 1, 1) AS CDPCSP
            FROM (
                SELECT 
                    ID_DWR_CLI_CIAL,
                    CDPCSP,
                    ROW_NUMBER() OVER (
                        PARTITION BY ID_DWR_CLI_CIAL 
                        ORDER BY CASE
                            WHEN CDPCSP = '3' THEN 1
                            WHEN CDPCSP = '2' THEN 2
                            ELSE 9
                        END
                    ) AS row_num
                FROM {db}.connaissance_client
            ) tmp
            WHERE row_num = 1
        ),
        BASE AS (
            SELECT 
                ID_DWR_CLI_CIAL,
                CAST(MAX(AGE) AS INT) as AGE,
                CAST(MAX(FRONTALIER) AS INT) as FRONTALIER,
                CAST(MAX(nb_majeur) AS INT) as nb_majeur,
                CAST(MAX(nb_enfant) AS INT) as nb_enfant
            FROM {db}.connaissance_client
            WHERE AGE >= {age_min} AND AGE <= {age_max} 
              AND nb_majeur >= {nb_majeur_min} AND AGE IS NOT NULL
            GROUP BY ID_DWR_CLI_CIAL
        )
        SELECT 
            BASE.ID_DWR_CLI_CIAL,
            BASE.AGE, BASE.FRONTALIER, BASE.nb_majeur, BASE.nb_enfant,
            COALESCE(LGT.CD_OCC_LGT, 'NC') AS CD_OCC_LGT,
            COALESCE(SIFA.CDSIFA, 'Non renseigné') AS CDSIFA,
            COALESCE(CSP.CDPCSP, '9') AS CDPCSP
        FROM BASE
        LEFT JOIN LGT ON BASE.ID_DWR_CLI_CIAL = LGT.ID_DWR_CLI_CIAL
        LEFT JOIN SIFA ON BASE.ID_DWR_CLI_CIAL = SIFA.ID_DWR_CLI_CIAL
        LEFT JOIN CSP ON BASE.ID_DWR_CLI_CIAL = CSP.ID_DWR_CLI_CIAL
        """,
        
        'organizational': """
        SELECT 
            ID_DWR_CLI_CIAL,
            num_dsc, nom_dsc, num_pole, nom_pole, num_agence, nom_agence
        FROM (
            SELECT 
                ID_DWR_CLI_CIAL,
                COALESCE(TRIM(num_dsc), 'NC') as num_dsc,
                COALESCE(TRIM(nom_dsc), 'Non renseigné') as nom_dsc,
                COALESCE(TRIM(num_pole), 'NC') as num_pole,
                COALESCE(TRIM(nom_pole), 'Non renseigné') as nom_pole,
                COALESCE(TRIM(num_agence), 'NC') as num_agence,
                COALESCE(TRIM(nom_agence), 'Non renseigné') as nom_agence,
                ROW_NUMBER() OVER (
                    PARTITION BY ID_DWR_CLI_CIAL 
                    ORDER BY 
                        CASE WHEN num_agence IS NOT NULL AND num_agence != '' THEN 1 ELSE 2 END,
                        CASE WHEN num_pole IS NOT NULL AND num_pole != '' THEN 1 ELSE 2 END
                ) AS row_num
            FROM {db}.connaissance_client
            WHERE ID_DWR_CLI_CIAL IS NOT NULL
        ) ranked
        WHERE row_num = 1
        """,
        
        'financial_pnb': """
        SELECT 
            ID_DWR_CLI_CIAL,
            CD_NOTE_UNVRS_ASSU, CD_NOTE_UNVRS_CRED, CD_NOTE_UNVRS_EPRGN, 
            CD_NOTE_UNVRS_SERV, CD_NOTE_POIDS_UNVRS, CD_NOTE_MIRE,
            SUM(MT_PNB_ASSU_AGLS) as PNB_ASSU,
            SUM(MT_PNB_COLCT_AGLS) as PNB_COLL,
            SUM(MT_PNB_CRED_AGLS) + SUM(MT_PNB_ADI_AGLS) as PNB_CRED,
            SUM(MT_PNB_SERV_FACTN_AGLS) as PNB_SERV
        FROM {db}.connaissance_client
        GROUP BY ID_DWR_CLI_CIAL, CD_NOTE_UNVRS_ASSU, CD_NOTE_UNVRS_CRED, 
            CD_NOTE_UNVRS_EPRGN, CD_NOTE_UNVRS_SERV, CD_NOTE_POIDS_UNVRS, CD_NOTE_MIRE
        """,
        
        'financial_patrimoine': """
        SELECT 
            RTRIM(baq.ID_DWR_CLI_CIAL) as ID_DWR_CLI_CIAL,
            COALESCE(a, 0) as SLD_DAV,
            ROUND(COALESCE(liquide, 0), 2) AS EP_LIQUIDE,
            ROUND(COALESCE(b, 0) + COALESCE(stable, 0), 2) AS EP_STABLE
        FROM (
            SELECT ID_DWR_CLI_CIAL, SUM(SLD_MOY_12_MOIS) AS A 
            FROM {db}.baq GROUP BY ID_DWR_CLI_CIAL
        ) as baq
        LEFT JOIN (
            SELECT ID_DWR_CLI_CIAL, SUM(MT_ENCRS) AS B 
            FROM {db}.epargne_assurance GROUP BY ID_DWR_CLI_CIAL
        ) as epargne_assurance ON baq.ID_DWR_CLI_CIAL = epargne_assurance.ID_DWR_CLI_CIAL
        LEFT JOIN (
            SELECT ID_DWR_CLI_CIAL,
                SUM(CASE WHEN fichier_source = 'FEPAVE02' THEN MONTANT ELSE 0 END) AS liquide,
                SUM(CASE WHEN fichier_source <> 'FEPAVE02' THEN MONTANT ELSE 0 END) AS stable
            FROM {db}.epargne GROUP BY ID_DWR_CLI_CIAL
        ) as epargne ON epargne.ID_DWR_CLI_CIAL = baq.ID_DWR_CLI_CIAL
        """,
        
        'financial_rfm': """
        SELECT 
            ID_DWR_CLI_CIAL,
            SUM(NB_OPERN_DBTR_MOIS) AS frequency,
            AVG(FLUX_MENS) AS amount,
            MIN(RECENCY) AS recency
        FROM {db}.baq
        GROUP BY ID_DWR_CLI_CIAL
        """
    }
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.spark = spark
        self.database = BaseDonnees if BaseDonnees else Config.DATABASE

        # Logger unifié
        self.logger = get_logger('data_extractor')

        # Gestionnaire de cache avec validation mois/année
        self.cache_manager = CacheManager(self.logger)
        self.current_period = get_current_period()

        self.logger.info(f"DataExtractor initialisé - Cache: {use_cache}, Période: {self.current_period}")
    
    def _execute_sql(self, sql_query: str) -> pd.DataFrame:
        """Exécute une requête SQL et retourne un DataFrame pandas"""
        if not SPARK_AVAILABLE:
            self.logger.error("Spark non disponible")
            return pd.DataFrame()
        
        try:
            df_spark = self.spark.sql(sql_query)
            df = df_spark.toPandas()
            return df
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution SQL : {e}")
            return pd.DataFrame()
    
    def extract_core_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Extrait les données socles avec cache intelligent mois/année.

        Le cache est réutilisé seulement si:
        - use_cache=True
        - force_refresh=False
        - Le cache existe et correspond au mois/année courant
        """
        cache_path = str(Config.get_cache_path('core_data'))

        # Tentative de chargement du cache
        if self.use_cache and not force_refresh:
            data, status = self.cache_manager.load_cache(cache_path, self.current_period)
            if data is not None:
                return data
            # Si cache invalide/périmé, on continue avec l'extraction
            self.logger.info(f"Cache non utilisable ({status}) - Extraction depuis la base")

        self.logger.info("Extraction données socles depuis la base")

        sql = self.SQL_QUERIES['core'].format(
            db=self.database,
            age_min=Config.EXTRACTION['age_min'],
            age_max=Config.EXTRACTION['age_max'],
            nb_majeur_min=Config.EXTRACTION['nb_majeur_min']
        )

        df = self._execute_sql(sql)

        if not df.empty:
            # Conversion des types
            int_cols = ['AGE', 'FRONTALIER', 'nb_majeur', 'nb_enfant']
            for col in int_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float).astype(pd.Int64Dtype())

            # Sauvegarde cache avec période
            if self.use_cache:
                self.cache_manager.save_cache(cache_path, df, self.current_period)

            self.logger.info(f"Extraction socles terminée : {len(df):,} lignes")

        return df
    
    def extract_organizational_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Extrait les données organisationnelles avec cache intelligent mois/année.
        """
        cache_path = str(Config.get_cache_path('organizational_data'))

        # Tentative de chargement du cache
        if self.use_cache and not force_refresh:
            data, status = self.cache_manager.load_cache(cache_path, self.current_period)
            if data is not None:
                return data
            self.logger.info(f"Cache non utilisable ({status}) - Extraction depuis la base")

        self.logger.info("Extraction données organisationnelles depuis la base")

        sql = self.SQL_QUERIES['organizational'].format(db=self.database)
        df = self._execute_sql(sql)

        if not df.empty:
            if self.use_cache:
                self.cache_manager.save_cache(cache_path, df, self.current_period)

            self.logger.info(f"Extraction organisationnelle terminée : {len(df):,} lignes")

        return df
    
    def extract_financial_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Extrait les données financières avec cache intelligent mois/année.
        """
        cache_path = str(Config.get_cache_path('financial_data'))

        # Tentative de chargement du cache
        if self.use_cache and not force_refresh:
            data, status = self.cache_manager.load_cache(cache_path, self.current_period)
            if data is not None:
                return data
            self.logger.info(f"Cache non utilisable ({status}) - Extraction depuis la base")

        self.logger.info("Extraction données financières depuis la base")

        # Extraction des différentes composantes
        df_pnb = self._execute_sql(self.SQL_QUERIES['financial_pnb'].format(db=self.database))
        df_pat = self._execute_sql(self.SQL_QUERIES['financial_patrimoine'].format(db=self.database))
        df_rfm = self._execute_sql(self.SQL_QUERIES['financial_rfm'].format(db=self.database))

        # Fusion
        df = df_pnb
        if not df_pat.empty:
            df = df.merge(df_pat, on=Config.EXTRACTION['id_column'], how='outer')
        if not df_rfm.empty:
            df = df.merge(df_rfm, on=Config.EXTRACTION['id_column'], how='outer')

        if not df.empty:
            df = df.fillna(0)

            # Conversion des types
            float_cols = ['SLD_DAV', 'EP_LIQUIDE', 'EP_STABLE', 'PNB_ASSU', 'PNB_COLL', 'PNB_CRED', 'PNB_SERV', 'amount']
            for col in float_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            int_cols = ['frequency', 'recency']
            for col in int_cols:
                if col in df.columns:
                    df[col] = df[col].astype(int)

            if self.use_cache:
                self.cache_manager.save_cache(cache_path, df, self.current_period)

            self.logger.info(f"Extraction financière terminée : {len(df):,} lignes")

        return df
    
    def extract_all_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Extrait et fusionne toutes les données"""
        self.logger.info("Début extraction complète")
        
        # Extraction des différentes composantes
        df_core = self.extract_core_data(force_refresh)
        df_org = self.extract_organizational_data(force_refresh)
        df_fin = self.extract_financial_data(force_refresh)
        
        # Fusion progressive
        df_final = df_core.copy()
        
        if not df_org.empty:
            df_final = df_final.merge(df_org, on=Config.EXTRACTION['id_column'], how='left')
            self.logger.info("Données organisationnelles fusionnées")
        
        if not df_fin.empty:
            df_final = df_final.merge(df_fin, on=Config.EXTRACTION['id_column'], how='left')
            self.logger.info("Données financières fusionnées")
        
        self.logger.info(f"Extraction complète terminée : {len(df_final):,} lignes, {len(df_final.columns)} colonnes")
        
        return df_final
    
    def get_sample(self, n: int = 10000, random_state: int = 42) -> pd.DataFrame:
        """Récupère un échantillon des données complètes"""
        df = self.extract_all_data()
        if df.empty:
            return df
        return df.sample(n=min(n, len(df)), random_state=random_state)
    
    def clear_cache(self, data_type: Optional[str] = None):
        """Vide le cache (tout ou un type spécifique)"""
        if data_type and data_type in Config.CACHE_FILES:
            cache_path = Config.get_cache_path(data_type)
            if cache_path.exists():
                cache_path.unlink()
                self.logger.info(f"Cache {data_type} vidé")
        else:
            for cache_type in Config.CACHE_FILES.keys():
                cache_path = Config.get_cache_path(cache_type)
                if cache_path.exists():
                    cache_path.unlink()
            self.logger.info("Tous les caches vidés")


# Fonctions utilitaires
def extract_data(sample_size: Optional[int] = None, use_cache: bool = True) -> pd.DataFrame:
    """Fonction utilitaire simplifiée pour extraction rapide"""
    extractor = DataExtractor(use_cache=use_cache)
    if sample_size:
        return extractor.get_sample(sample_size)
    return extractor.extract_all_data()
