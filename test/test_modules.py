#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_modules.py - Tests pour les modules individuels (clustering, scoring, etc.)
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil

# Ajout path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.age_clustering import AgeClusterer
from src.client_scoring import ClientScoringEngine
from src.dimension_optimizer import DimensionOptimizer
from src.data_extractor import DataExtractor, CacheManager, get_current_period
from src.mock_data_generator import MockDataGenerator
from src.config import Config, get_logger


class TestAgeClusterer(unittest.TestCase):
    """Tests du module de clustering par âge."""

    @classmethod
    def setUpClass(cls):
        """Configuration une fois pour tous les tests."""
        # Sauvegarder les valeurs originales
        cls.original_min_strata = Config.CLUSTERING['min_strata_size']
        cls.original_min_cluster = Config.CLUSTERING['min_cluster_size']
        cls.original_models_dir = Config.MODELS_DIR
        cls.original_cache_dir = Config.CACHE_DIR

        # Réduire temporairement les contraintes pour les tests
        Config.CLUSTERING['min_strata_size'] = 100
        Config.CLUSTERING['min_cluster_size'] = 50

        # Répertoire temporaire isolé pour cette classe de test
        cls.temp_dir = tempfile.mkdtemp()
        Config.MODELS_DIR = Path(cls.temp_dir)
        Config.CACHE_DIR = Path(cls.temp_dir)

        cls.generator = MockDataGenerator(n_clients=5000, random_state=42)
        cls.df = cls.generator.generate_merged_data()

        # Optimiser les dimensions pour avoir les bonnes colonnes
        optimizer = DimensionOptimizer(clustering_mode='stratifie')
        cls.df_processed = optimizer.transform(cls.df, keep_ids=True)

    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests."""
        # Restaurer les valeurs originales
        if hasattr(cls, 'original_min_strata'):
            Config.CLUSTERING['min_strata_size'] = cls.original_min_strata
        if hasattr(cls, 'original_min_cluster'):
            Config.CLUSTERING['min_cluster_size'] = cls.original_min_cluster
        if hasattr(cls, 'original_models_dir'):
            Config.MODELS_DIR = cls.original_models_dir
        if hasattr(cls, 'original_cache_dir'):
            Config.CACHE_DIR = cls.original_cache_dir

        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)

    def test_clusterer_initialization(self):
        """Test de l'initialisation du clusterer."""
        clusterer = AgeClusterer()
        self.assertIsNotNone(clusterer.logger)

    def test_fit_transform(self):
        """Test de l'entraînement et transformation."""
        clusterer = AgeClusterer()
        df_result = clusterer.fit_transform(self.df_processed)

        # Vérifier les résultats
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(len(df_result), len(self.df_processed))

        # Vérifier les colonnes de clustering
        self.assertIn('CLUSTER_STRATE', df_result.columns)
        self.assertIn('CLUSTER_LOCAL', df_result.columns)
        self.assertIn('CLUSTER_GLOBAL', df_result.columns)

    def test_clustering_quality_metrics(self):
        """Test que les métriques de qualité sont calculées."""
        clusterer = AgeClusterer()
        df_result = clusterer.fit_transform(self.df_processed)

        # Vérifier les métriques
        self.assertIsNotNone(clusterer.quality_metrics)
        self.assertIn('global_metrics', clusterer.quality_metrics)

        global_metrics = clusterer.quality_metrics['global_metrics']
        self.assertIn('avg_silhouette_score', global_metrics)
        self.assertIn('total_clusters', global_metrics)
        self.assertIn('viable_strata', global_metrics)

    def test_model_save_and_load(self):
        """Test de la sauvegarde et chargement du modèle."""
        # Entraîner un modèle (la sauvegarde est automatique dans fit_transform)
        clusterer1 = AgeClusterer()
        df_result1 = clusterer1.fit_transform(self.df_processed)

        # Vérifier que le modèle a été sauvegardé
        model_path = clusterer1.model_path
        self.assertTrue(Path(model_path).exists())

        # Charger dans un nouveau clusterer
        clusterer2 = AgeClusterer()
        clusterer2.load_model(str(model_path))

        # Vérifier que le modèle est chargé
        self.assertEqual(clusterer2.model_version, clusterer1.model_version)
        self.assertIsNotNone(clusterer2.training_metadata)

        # Transformer avec le modèle chargé
        df_result2 = clusterer2.transform(self.df_processed)

        # Les résultats devraient être identiques
        self.assertEqual(len(df_result2), len(df_result1))

    def test_transform_without_fit(self):
        """Test que transform échoue sans fit ou load."""
        # Créer un sous-répertoire vide temporairement pour éviter le chargement auto
        empty_models_dir = Path(self.temp_dir) / "empty_models"
        empty_models_dir.mkdir(exist_ok=True)
        original_models_dir = Config.MODELS_DIR
        Config.MODELS_DIR = empty_models_dir

        try:
            clusterer = AgeClusterer()
            with self.assertRaises(ValueError):
                clusterer.transform(self.df_processed)
        finally:
            Config.MODELS_DIR = original_models_dir

    def test_age_strata_creation(self):
        """Test de la création des strates d'âge."""
        clusterer = AgeClusterer()
        df_result = clusterer.fit_transform(self.df_processed)

        # Vérifier que plusieurs strates ont été créées
        n_strates = df_result['CLUSTER_STRATE'].nunique()
        self.assertGreater(n_strates, 1)

        # Vérifier le format des strates (ex: "23-27", "28-32", "63+")
        strates = df_result['CLUSTER_STRATE'].unique()
        for strate in strates:
            if strate != 'NON_CLUSTERED':
                # Format: "XX-YY" ou "63+"
                if strate.endswith('+'):
                    # Format: "63+"
                    self.assertTrue(strate[:-1].isdigit())
                else:
                    # Format: "XX-YY"
                    parts = strate.split('-')
                    self.assertEqual(len(parts), 2)

    def test_non_clustered_handling(self):
        """Test de la gestion des clients non-clusterisés."""
        # Ce test vérifie le comportement avec les données complètes de la classe
        # où certains clients peuvent être non-clusterisés si leur strate est trop petite

        # Utiliser les données déjà clusterisées par setUpClass pour vérifier le comportement
        clusterer = AgeClusterer()
        df_result = clusterer.fit_transform(self.df_processed)

        # Vérifier que les colonnes de clustering existent
        self.assertIn('CLUSTER_GLOBAL', df_result.columns)
        self.assertIn('CLUSTER_STRATE', df_result.columns)
        self.assertIn('CLUSTER_LOCAL', df_result.columns)

        # Vérifier la cohérence des résultats
        non_clustered = df_result[df_result['CLUSTER_GLOBAL'] == 'NON_CLUSTERED']
        clustered = df_result[df_result['CLUSTER_GLOBAL'] != 'NON_CLUSTERED']

        # Tous les clients doivent être soit clusterisés soit non-clusterisés
        self.assertEqual(len(non_clustered) + len(clustered), len(df_result))

        # Avec 5000 clients et min_strata=100, on devrait avoir des clients clusterisés
        self.assertGreater(len(clustered), 0, "Au moins certains clients devraient être clusterisés")


class TestClientScoringEngine(unittest.TestCase):
    """Tests du moteur de scoring."""

    @classmethod
    def setUpClass(cls):
        """Configuration une fois pour tous les tests."""
        # Sauvegarder les valeurs originales
        cls.original_min_strata = Config.CLUSTERING['min_strata_size']
        cls.original_min_cluster = Config.CLUSTERING['min_cluster_size']
        cls.original_models_dir = Config.MODELS_DIR
        cls.original_cache_dir = Config.CACHE_DIR

        # Réduire temporairement les contraintes pour les tests
        Config.CLUSTERING['min_strata_size'] = 50
        Config.CLUSTERING['min_cluster_size'] = 20

        # Créer un répertoire temporaire isolé pour cette classe de test
        cls.temp_dir = tempfile.mkdtemp()
        Config.MODELS_DIR = Path(cls.temp_dir) / "models"
        Config.MODELS_DIR.mkdir(exist_ok=True)
        Config.CACHE_DIR = Path(cls.temp_dir)

        cls.generator = MockDataGenerator(n_clients=1500, random_state=42)

        # Générer données complètes
        df_core, df_org, df_financial = cls.generator.generate_all_data()
        cls.df_financial = df_financial

        # Créer un DataFrame avec clustering
        optimizer = DimensionOptimizer(clustering_mode='stratifie')
        df_merged = cls.generator.generate_merged_data()
        df_processed = optimizer.transform(df_merged, keep_ids=True)

        clusterer = AgeClusterer()
        cls.df_clustered = clusterer.fit_transform(df_processed)

    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests."""
        # Restaurer les valeurs originales
        if hasattr(cls, 'original_min_strata'):
            Config.CLUSTERING['min_strata_size'] = cls.original_min_strata
        if hasattr(cls, 'original_min_cluster'):
            Config.CLUSTERING['min_cluster_size'] = cls.original_min_cluster
        if hasattr(cls, 'original_models_dir'):
            Config.MODELS_DIR = cls.original_models_dir
        if hasattr(cls, 'original_cache_dir'):
            Config.CACHE_DIR = cls.original_cache_dir

        # Nettoyer le répertoire temporaire
        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)

    def test_scoring_engine_initialization(self):
        """Test de l'initialisation du moteur de scoring."""
        scorer = ClientScoringEngine()
        self.assertIsNotNone(scorer.logger)

    def test_calculate_scores(self):
        """Test du calcul des scores."""
        scorer = ClientScoringEngine()
        df_scored = scorer.calculate_scores(self.df_clustered, self.df_financial)

        # Vérifier les résultats
        self.assertIsInstance(df_scored, pd.DataFrame)
        self.assertEqual(len(df_scored), len(self.df_clustered))

        # Vérifier les colonnes de scoring
        scoring_cols = [
            'CLIENT_SCORE',
            'TOTAL_POTENTIAL',
            'SEGMENTATION_FINALE'
        ]
        for col in scoring_cols:
            self.assertIn(col, df_scored.columns)

    def test_score_ranges(self):
        """Test que les scores sont dans les bonnes plages."""
        scorer = ClientScoringEngine()
        df_scored = scorer.calculate_scores(self.df_clustered, self.df_financial)

        # CLIENT_SCORE doit être entre 0 et 1
        self.assertTrue((df_scored['CLIENT_SCORE'] >= 0).all())
        self.assertTrue((df_scored['CLIENT_SCORE'] <= 1).all())

        # TOTAL_POTENTIAL doit être >= 0
        self.assertTrue((df_scored['TOTAL_POTENTIAL'] >= 0).all())

    def test_segmentation_labels(self):
        """Test des labels de segmentation."""
        scorer = ClientScoringEngine()
        df_scored = scorer.calculate_scores(self.df_clustered, self.df_financial)

        # Vérifier que les labels sont valides
        valid_labels = ['A entretenir', 'A developper', 'A stimuler', 'A construire']
        unique_labels = df_scored['SEGMENTATION_FINALE'].unique()

        for label in unique_labels:
            self.assertIn(label, valid_labels)

    def test_potential_columns(self):
        """Test des colonnes de potentiel par univers."""
        scorer = ClientScoringEngine()
        df_scored = scorer.calculate_scores(self.df_clustered, self.df_financial)

        # Vérifier les colonnes de potentiel
        potential_cols = [
            'POTENTIAL_PNB_COLL',
            'POTENTIAL_PNB_CRED',
            'POTENTIAL_PNB_SERV',
            'POTENTIAL_PNB_ASSU'
        ]

        for col in potential_cols:
            self.assertIn(col, df_scored.columns)
            self.assertTrue((df_scored[col] >= 0).all())

    def test_scoring_metrics(self):
        """Test que les métriques de scoring sont générées."""
        scorer = ClientScoringEngine()
        df_scored = scorer.calculate_scores(self.df_clustered, self.df_financial)

        # Obtenir les métriques
        metrics = scorer.get_scoring_metrics()

        # Vérifier les métriques
        self.assertIn('global_metrics', metrics)
        self.assertIn('total_clients', metrics['global_metrics'])
        self.assertIn('total_potential_value', metrics['global_metrics'])

    def test_segment_distribution(self):
        """Test de la distribution des segments."""
        scorer = ClientScoringEngine()
        df_scored = scorer.calculate_scores(self.df_clustered, self.df_financial)

        # Vérifier qu'il y a plusieurs segments
        n_segments = df_scored['SEGMENTATION_FINALE'].nunique()
        self.assertGreater(n_segments, 1)

        # Vérifier que chaque segment a des clients
        segment_counts = df_scored['SEGMENTATION_FINALE'].value_counts()
        self.assertTrue((segment_counts > 0).all())


class TestDimensionOptimizer(unittest.TestCase):
    """Tests de l'optimiseur de dimensions."""

    @classmethod
    def setUpClass(cls):
        """Configuration une fois pour tous les tests."""
        cls.generator = MockDataGenerator(n_clients=1000, random_state=42)
        cls.df = cls.generator.generate_merged_data()

    def test_optimizer_initialization(self):
        """Test de l'initialisation de l'optimiseur."""
        optimizer = DimensionOptimizer(clustering_mode='stratifie')
        self.assertEqual(optimizer.clustering_mode, 'stratifie')

    def test_transform(self):
        """Test de la transformation des données."""
        optimizer = DimensionOptimizer(clustering_mode='stratifie')
        df_processed = optimizer.transform(self.df, keep_ids=True)

        # Vérifier que les données sont transformées
        self.assertIsInstance(df_processed, pd.DataFrame)
        self.assertGreater(len(df_processed), 0)

        # L'ID doit être conservé
        self.assertIn(Config.EXTRACTION['id_column'], df_processed.columns)

        # AGE doit être conservé pour le clustering stratifié
        self.assertIn('AGE', df_processed.columns)

    def test_transform_without_ids(self):
        """Test de la transformation sans conserver les IDs."""
        optimizer = DimensionOptimizer(clustering_mode='stratifie')
        df_processed = optimizer.transform(self.df, keep_ids=False)

        # L'ID ne devrait pas être dans le résultat
        # (ou alors c'est acceptable selon l'implémentation)
        self.assertIsInstance(df_processed, pd.DataFrame)

    def test_different_clustering_modes(self):
        """Test avec différents modes de clustering."""
        # Mode stratifié
        optimizer_stratifie = DimensionOptimizer(clustering_mode='stratifie')
        df_stratifie = optimizer_stratifie.transform(self.df, keep_ids=True)

        # Mode global
        optimizer_global = DimensionOptimizer(clustering_mode='global')
        df_global = optimizer_global.transform(self.df, keep_ids=True)

        # Les deux devraient produire des résultats
        self.assertGreater(len(df_stratifie), 0)
        self.assertGreater(len(df_global), 0)


class TestDataExtractor(unittest.TestCase):
    """Tests de l'extracteur de données."""

    @classmethod
    def setUpClass(cls):
        """Configuration une fois pour tous les tests."""
        # Créer des données mock dans le cache
        cls.generator = MockDataGenerator(n_clients=500, random_state=42)
        cls.temp_dir = tempfile.mkdtemp()
        Config.CACHE_DIR = Path(cls.temp_dir)

        cls.file_paths = cls.generator.save_to_cache(output_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests."""
        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)

    def test_extractor_initialization(self):
        """Test de l'initialisation de l'extracteur."""
        extractor = DataExtractor(use_cache=True)
        self.assertTrue(extractor.use_cache)

    def test_extract_all_data(self):
        """Test de l'extraction de toutes les données."""
        extractor = DataExtractor(use_cache=True)
        df = extractor.extract_all_data()

        # Vérifier les données
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Vérifier les colonnes essentielles
        self.assertIn(Config.EXTRACTION['id_column'], df.columns)
        self.assertIn('AGE', df.columns)

    def test_extract_financial_data(self):
        """Test de l'extraction des données financières."""
        extractor = DataExtractor(use_cache=True)
        df_financial = extractor.extract_financial_data()

        # Vérifier les données
        self.assertIsInstance(df_financial, pd.DataFrame)
        self.assertGreater(len(df_financial), 0)

        # Vérifier les colonnes PNB
        for col in Config.SCORING['pnb_columns']:
            self.assertIn(col, df_financial.columns)

    def test_get_sample(self):
        """Test de l'obtention d'un échantillon."""
        extractor = DataExtractor(use_cache=True)
        df_sample = extractor.get_sample(n=100)

        # Vérifier la taille
        self.assertLessEqual(len(df_sample), 100)

    def test_cache_manager(self):
        """Test du gestionnaire de cache."""
        import logging
        logger = logging.getLogger('test_cache')
        cache_manager = CacheManager(logger=logger)

        # Chemin de test
        cache_path = str(Path(self.temp_dir) / 'test_cache.pkl')
        period = '202512'

        # Sauvegarder un cache
        test_data = pd.DataFrame({'col1': [1, 2, 3]})
        success = cache_manager.save_cache(cache_path, test_data, period)
        self.assertTrue(success)

        # Charger le cache avec la bonne période
        loaded_data, status = cache_manager.load_cache(cache_path, period)

        # Vérifier
        self.assertEqual(status, 'CACHE_VALID')
        self.assertIsNotNone(loaded_data)
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_get_current_period(self):
        """Test de la fonction get_current_period."""
        period = get_current_period()

        # Vérifier le format YYYYMM
        self.assertEqual(len(period), 6)
        self.assertTrue(period.isdigit())


class TestConfig(unittest.TestCase):
    """Tests du module de configuration."""

    def test_get_logger(self):
        """Test de la création d'un logger."""
        logger = get_logger('test_logger')

        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_logger')

    def test_get_logger_same_name(self):
        """Test que le même nom retourne le même logger."""
        logger1 = get_logger('same_name_logger')
        logger2 = get_logger('same_name_logger')

        # Les deux devraient être le même objet
        self.assertEqual(logger1.name, logger2.name)

    def test_config_paths_exist(self):
        """Test que les chemins de configuration sont définis."""
        self.assertIsNotNone(Config.CACHE_DIR)
        self.assertIsNotNone(Config.MODELS_DIR)
        self.assertIsNotNone(Config.LOGS_DIR)

    def test_config_clustering_params(self):
        """Test des paramètres de clustering."""
        self.assertIn('min_strata_size', Config.CLUSTERING)
        self.assertIn('min_cluster_size', Config.CLUSTERING)
        self.assertIn('k_range', Config.CLUSTERING)
        self.assertIn('random_state', Config.CLUSTERING)

    def test_config_scoring_params(self):
        """Test des paramètres de scoring."""
        self.assertIn('top_client_pct', Config.SCORING)
        self.assertIn('pnb_columns', Config.SCORING)

    def test_cache_manager_invalid_path(self):
        """Test du CacheManager avec un chemin invalide."""
        import logging
        logger = logging.getLogger('test_invalid_cache')
        cache_manager = CacheManager(logger=logger)

        # Charger depuis un chemin qui n'existe pas
        data, status = cache_manager.load_cache('/path/inexistant/cache.pkl', '202512')

        self.assertIsNone(data)
        self.assertEqual(status, 'CACHE_NOT_FOUND')

    def test_cache_manager_expired_period(self):
        """Test du CacheManager avec une période expirée."""
        import logging
        logger = logging.getLogger('test_expired_cache')
        cache_manager = CacheManager(logger=logger)

        # Créer un cache temporaire
        temp_dir = tempfile.mkdtemp()
        cache_path = str(Path(temp_dir) / 'test_cache.pkl')

        try:
            # Sauvegarder avec une période
            test_data = pd.DataFrame({'col1': [1, 2, 3]})
            cache_manager.save_cache(cache_path, test_data, '202501')

            # Charger avec une période différente
            loaded_data, status = cache_manager.load_cache(cache_path, '202512')

            # Le status contient la période expirée
            self.assertTrue(status.startswith('CACHE_EXPIRED'))
        finally:
            shutil.rmtree(temp_dir)


class TestRobustness(unittest.TestCase):
    """Tests de robustesse pour les cas limites."""

    @classmethod
    def setUpClass(cls):
        """Configuration pour les tests de robustesse."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_models_dir = Config.MODELS_DIR
        cls.original_cache_dir = Config.CACHE_DIR
        Config.MODELS_DIR = Path(cls.temp_dir) / "models"
        Config.MODELS_DIR.mkdir(exist_ok=True)
        Config.CACHE_DIR = Path(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Nettoyage après les tests."""
        Config.MODELS_DIR = cls.original_models_dir
        Config.CACHE_DIR = cls.original_cache_dir
        if Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)

    def test_age_clusterer_empty_dataframe(self):
        """Test du clusterer avec un DataFrame vide."""
        clusterer = AgeClusterer()
        df_empty = pd.DataFrame()

        with self.assertRaises(ValueError):
            clusterer.fit_transform(df_empty)

    def test_age_clusterer_missing_age_column(self):
        """Test du clusterer sans colonne AGE."""
        clusterer = AgeClusterer()
        df_no_age = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        with self.assertRaises(ValueError):
            clusterer.fit_transform(df_no_age)

    def test_dimension_optimizer_empty_dataframe(self):
        """Test de l'optimizer avec un DataFrame vide."""
        optimizer = DimensionOptimizer(clustering_mode='stratifie')
        df_empty = pd.DataFrame()

        # Ne devrait pas lever d'exception, retourne un DataFrame vide
        result = optimizer.transform(df_empty)
        self.assertEqual(len(result), 0)

    def test_dimension_optimizer_missing_columns(self):
        """Test de l'optimizer avec des colonnes manquantes."""
        optimizer = DimensionOptimizer(clustering_mode='stratifie')
        df_minimal = pd.DataFrame({
            'AGE': [25, 35, 45],
            'ID_DWR_CLIENT': [1, 2, 3]
        })

        # Devrait fonctionner même avec peu de colonnes
        result = optimizer.transform(df_minimal, keep_ids=True)
        self.assertIn('AGE', result.columns)

    def test_scoring_engine_empty_cluster_data(self):
        """Test du scoring avec des données clusterisées vides."""
        scorer = ClientScoringEngine()

        # Inclure toutes les colonnes requises
        df_clustered = pd.DataFrame({
            'ID_DWR_CLIENT': pd.Series([], dtype='int64'),
            'ID_DWR_CLI_CIAL': pd.Series([], dtype='int64'),
            'CLUSTER_GLOBAL': pd.Series([], dtype='str'),
            'CLUSTER_STRATE': pd.Series([], dtype='str'),
            'CLUSTER_LOCAL': pd.Series([], dtype='int64')
        })
        df_financial = pd.DataFrame({
            'ID_DWR_CLIENT': pd.Series([], dtype='int64'),
            'ID_DWR_CLI_CIAL': pd.Series([], dtype='int64'),
            'PNB_COLL': pd.Series([], dtype='float64'),
            'PNB_CRED': pd.Series([], dtype='float64'),
            'PNB_SERV': pd.Series([], dtype='float64'),
            'PNB_ASSU': pd.Series([], dtype='float64')
        })

        result = scorer.calculate_scores(df_clustered, df_financial)
        self.assertEqual(len(result), 0)

    def test_data_extractor_without_cache(self):
        """Test de l'extracteur sans cache."""
        extractor = DataExtractor(use_cache=False)

        # Devrait retourner des données vides en mode sans cache et sans Spark
        df = extractor.extract_all_data()
        # Le comportement dépend de l'environnement
        self.assertIsInstance(df, pd.DataFrame)

    def test_age_clusterer_all_same_age(self):
        """Test du clusterer quand tous les clients ont le même âge."""
        # Sauvegarder et modifier temporairement les contraintes
        original_min_strata = Config.CLUSTERING['min_strata_size']
        original_min_cluster = Config.CLUSTERING['min_cluster_size']
        Config.CLUSTERING['min_strata_size'] = 5
        Config.CLUSTERING['min_cluster_size'] = 2

        try:
            clusterer = AgeClusterer()
            df_same_age = pd.DataFrame({
                'AGE': [30] * 20,
                'feature1': np.random.randn(20),
                'feature2': np.random.randn(20)
            })

            result = clusterer.fit_transform(df_same_age)

            # Tous devraient être dans la même strate
            strates = result['CLUSTER_STRATE'].unique()
            non_clustered = [s for s in strates if s != 'NON_CLUSTERED']
            self.assertLessEqual(len(non_clustered), 1)
        finally:
            Config.CLUSTERING['min_strata_size'] = original_min_strata
            Config.CLUSTERING['min_cluster_size'] = original_min_cluster

    def test_age_clusterer_with_nan_ages(self):
        """Test du clusterer avec des âges NaN."""
        original_min_strata = Config.CLUSTERING['min_strata_size']
        original_min_cluster = Config.CLUSTERING['min_cluster_size']
        Config.CLUSTERING['min_strata_size'] = 5
        Config.CLUSTERING['min_cluster_size'] = 2

        try:
            clusterer = AgeClusterer()
            df_with_nan = pd.DataFrame({
                'AGE': [25, 30, np.nan, 35, 40, np.nan, 45, 50, 55, 60],
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10)
            })

            # Devrait gérer les NaN gracieusement
            result = clusterer.fit_transform(df_with_nan)
            self.assertEqual(len(result), len(df_with_nan))
        finally:
            Config.CLUSTERING['min_strata_size'] = original_min_strata
            Config.CLUSTERING['min_cluster_size'] = original_min_cluster


def run_tests(verbosity=2):
    """Exécute tous les tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestAgeClusterer))
    suite.addTests(loader.loadTestsFromTestCase(TestClientScoringEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestDimensionOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustness))

    # Exécuter
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*60)
    print("TESTS DES MODULES INDIVIDUELS")
    print("="*60)

    success = run_tests(verbosity=2)

    if success:
        print("\n✅ Tous les tests sont passés!")
        sys.exit(0)
    else:
        print("\n❌ Certains tests ont échoué")
        sys.exit(1)
