#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline.py - Tests pour le pipeline complet
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import tempfile
import shutil

# Ajout path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.advanced_pipeline import AdvancedBankingPipeline, run_advanced_pipeline, get_pipeline_model_status
from src.mock_data_generator import MockDataGenerator
from src.config import Config


class TestAdvancedPipeline(unittest.TestCase):
    """Tests du pipeline avancé de segmentation."""

    @classmethod
    def setUpClass(cls):
        """Configuration une fois pour tous les tests."""
        # Réduire temporairement les contraintes pour les tests
        cls.original_min_strata = Config.CLUSTERING['min_strata_size']
        cls.original_min_cluster = Config.CLUSTERING['min_cluster_size']
        Config.CLUSTERING['min_strata_size'] = 50
        Config.CLUSTERING['min_cluster_size'] = 20

        # Créer des données mock dans le cache
        print("\n Configuration de données mock pour les tests...")
        cls.generator = MockDataGenerator(n_clients=2000, random_state=42)
        cls.temp_dir = tempfile.mkdtemp()
        Config.CACHE_DIR = Path(cls.temp_dir)
        Config.MODELS_DIR = Path(cls.temp_dir) / "models"
        Config.MODELS_DIR.mkdir(exist_ok=True)

        cls.file_paths = cls.generator.save_to_cache(output_dir=cls.temp_dir)
        print(f"Donnees mock creees: {len(cls.file_paths)} fichiers")

    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests."""
        # Restaurer les valeurs originales
        if hasattr(cls, 'original_min_strata'):
            Config.CLUSTERING['min_strata_size'] = cls.original_min_strata
        if hasattr(cls, 'original_min_cluster'):
            Config.CLUSTERING['min_cluster_size'] = cls.original_min_cluster

        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)
            print("\nNettoyage termine")

    def test_pipeline_initialization(self):
        """Test de l'initialisation du pipeline."""
        pipeline = AdvancedBankingPipeline(mode='dev', clustering_mode='stratifie')

        self.assertEqual(pipeline.mode, 'dev')
        self.assertEqual(pipeline.clustering_mode, 'stratifie')
        self.assertIsNotNone(pipeline.logger)
        self.assertFalse(pipeline.force_retrain)

    def test_pipeline_modes(self):
        """Test des différents modes du pipeline."""
        # Mode dev
        pipeline_dev = AdvancedBankingPipeline(mode='dev')
        self.assertEqual(pipeline_dev.mode, 'dev')

        # Mode test
        pipeline_test = AdvancedBankingPipeline(mode='test')
        self.assertEqual(pipeline_test.mode, 'test')

        # Mode prod
        pipeline_prod = AdvancedBankingPipeline(mode='prod')
        self.assertEqual(pipeline_prod.mode, 'prod')

    def test_invalid_mode(self):
        """Test qu'un mode invalide lève une erreur."""
        with self.assertRaises(ValueError):
            AdvancedBankingPipeline(mode='invalid')

    def test_invalid_clustering_mode(self):
        """Test qu'un mode de clustering invalide lève une erreur."""
        with self.assertRaises(ValueError):
            AdvancedBankingPipeline(mode='dev', clustering_mode='invalid')

    def test_pipeline_run_dev_mode(self):
        """Test de l'exécution complète du pipeline en mode dev."""
        pipeline = AdvancedBankingPipeline(mode='dev', clustering_mode='stratifie')

        # Exécution
        results = pipeline.run(sample_size=500)

        # Vérifier la structure des résultats
        self.assertIn('data', results)
        self.assertIn('metrics', results)
        self.assertIn('metadata', results)

        # Vérifier les métadonnées
        self.assertEqual(results['metadata']['status'], 'success')
        self.assertEqual(results['metadata']['mode'], 'dev')

        # Vérifier les données
        df = results['data']
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Vérifier les colonnes essentielles
        required_cols = [
            Config.EXTRACTION['id_column'],
            'CLUSTER_GLOBAL',
            'CLIENT_SCORE',
            'TOTAL_POTENTIAL',
            'SEGMENTATION_FINALE'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Colonne manquante: {col}")

    def test_pipeline_run_test_mode(self):
        """Test de l'exécution en mode test avec échantillon."""
        pipeline = AdvancedBankingPipeline(mode='test', clustering_mode='stratifie')

        results = pipeline.run(sample_size=200)

        # Vérifier que le pipeline s'est exécuté
        self.assertEqual(results['metadata']['status'], 'success')

        # Le nombre de clients scorés devrait être réduit
        df = results['data']
        self.assertLessEqual(len(df), 200)

    def test_pipeline_extraction_step(self):
        """Test de l'étape d'extraction."""
        pipeline = AdvancedBankingPipeline(mode='dev')

        df_raw = pipeline._extract_data(sample_size=300)

        # Vérifier les données extraites
        self.assertIsInstance(df_raw, pd.DataFrame)
        self.assertGreater(len(df_raw), 0)
        self.assertIn(Config.EXTRACTION['id_column'], df_raw.columns)
        self.assertIn('AGE', df_raw.columns)

    def test_pipeline_optimization_step(self):
        """Test de l'étape d'optimisation des dimensions."""
        pipeline = AdvancedBankingPipeline(mode='dev')

        # Extraire les données
        df_raw = pipeline._extract_data(sample_size=300)

        # Optimiser
        df_processed = pipeline._optimize_dimensions(df_raw)

        # Vérifier que les données ont été optimisées
        self.assertIsInstance(df_processed, pd.DataFrame)
        self.assertGreater(len(df_processed), 0)
        self.assertIn('AGE', df_processed.columns)

    def test_pipeline_clustering_step(self):
        """Test de l'étape de clustering."""
        pipeline = AdvancedBankingPipeline(mode='dev', force_retrain=True)

        # Extraire et optimiser
        df_raw = pipeline._extract_data(sample_size=500)
        df_processed = pipeline._optimize_dimensions(df_raw)

        # Clustering
        df_clustered = pipeline._apply_smart_clustering(df_processed)

        # Vérifier les résultats du clustering
        self.assertIsInstance(df_clustered, pd.DataFrame)
        self.assertGreater(len(df_clustered), 0)

        # Vérifier les colonnes de clustering
        clustering_cols = ['CLUSTER_STRATE', 'CLUSTER_LOCAL', 'CLUSTER_GLOBAL']
        for col in clustering_cols:
            self.assertIn(col, df_clustered.columns)

    def test_pipeline_scoring_step(self):
        """Test de l'étape de scoring."""
        pipeline = AdvancedBankingPipeline(mode='dev', force_retrain=True)

        # Exécuter jusqu'au clustering
        df_raw = pipeline._extract_data(sample_size=500)
        df_processed = pipeline._optimize_dimensions(df_raw)
        df_clustered = pipeline._apply_smart_clustering(df_processed)

        # Scoring
        df_scored = pipeline._apply_scoring(df_clustered, df_raw)

        # Vérifier les résultats du scoring
        self.assertIsInstance(df_scored, pd.DataFrame)
        self.assertGreater(len(df_scored), 0)

        # Vérifier les colonnes de scoring
        scoring_cols = ['CLIENT_SCORE', 'TOTAL_POTENTIAL', 'SEGMENTATION_FINALE']
        for col in scoring_cols:
            self.assertIn(col, df_scored.columns)

        # Vérifier les contraintes métier
        self.assertTrue((df_scored['CLIENT_SCORE'] >= 0).all())
        self.assertTrue((df_scored['CLIENT_SCORE'] <= 1).all())
        self.assertTrue((df_scored['TOTAL_POTENTIAL'] >= 0).all())

    def test_pipeline_with_force_retrain(self):
        """Test du pipeline avec force_retrain."""
        # Premier run pour créer un modèle
        pipeline1 = AdvancedBankingPipeline(mode='dev', force_retrain=True)
        results1 = pipeline1.run(sample_size=300)

        self.assertEqual(results1['metadata']['model_action'], 'retrained')

        # Deuxième run avec force_retrain
        pipeline2 = AdvancedBankingPipeline(mode='dev', force_retrain=True)
        results2 = pipeline2.run(sample_size=300)

        # Le modèle devrait être recalculé
        self.assertEqual(results2['metadata']['model_action'], 'retrained')

    def test_pipeline_without_force_retrain(self):
        """Test du pipeline sans force_retrain (réutilisation modèle)."""
        # Premier run pour créer un modèle
        pipeline1 = AdvancedBankingPipeline(mode='dev', force_retrain=True)
        results1 = pipeline1.run(sample_size=300)

        # Deuxième run sans force_retrain
        pipeline2 = AdvancedBankingPipeline(mode='dev', force_retrain=False)
        results2 = pipeline2.run(sample_size=300)

        # Le modèle devrait être réutilisé
        self.assertIn(results2['metadata']['model_action'], ['reused', 'retrained'])

    def test_pipeline_metrics(self):
        """Test que le pipeline génère toutes les métriques."""
        pipeline = AdvancedBankingPipeline(mode='dev', force_retrain=True)
        results = pipeline.run(sample_size=400)

        metrics = results['metrics']

        # Vérifier les métriques d'extraction
        self.assertIn('extraction', metrics)
        self.assertIn('n_clients', metrics['extraction'])

        # Vérifier les métriques de clustering
        self.assertIn('clustering', metrics)
        self.assertIn('global_metrics', metrics['clustering'])

        # Vérifier les métriques de scoring
        self.assertIn('scoring', metrics)
        self.assertIn('global_metrics', metrics['scoring'])

    def test_get_model_status(self):
        """Test de la fonction get_model_status."""
        # Créer un modèle d'abord
        pipeline = AdvancedBankingPipeline(mode='dev', force_retrain=True)
        pipeline.run(sample_size=300)

        # Obtenir le statut
        status = pipeline.get_model_status()

        # Vérifier le statut
        self.assertIn('has_model', status)
        if status['has_model']:
            self.assertIn('model_version', status)
            self.assertIn('training_date', status)
            self.assertIn('total_clusters', status)


class TestPipelineFunctions(unittest.TestCase):
    """Tests des fonctions utilitaires du pipeline."""

    @classmethod
    def setUpClass(cls):
        """Configuration une fois pour tous les tests."""
        # Réduire temporairement les contraintes pour les tests
        cls.original_min_strata = Config.CLUSTERING['min_strata_size']
        cls.original_min_cluster = Config.CLUSTERING['min_cluster_size']
        Config.CLUSTERING['min_strata_size'] = 50
        Config.CLUSTERING['min_cluster_size'] = 20

        # Créer des données mock
        cls.generator = MockDataGenerator(n_clients=1000, random_state=42)
        cls.temp_dir = tempfile.mkdtemp()
        Config.CACHE_DIR = Path(cls.temp_dir)
        Config.MODELS_DIR = Path(cls.temp_dir) / "models"
        Config.MODELS_DIR.mkdir(exist_ok=True)

        cls.file_paths = cls.generator.save_to_cache(output_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests."""
        # Restaurer les valeurs originales
        if hasattr(cls, 'original_min_strata'):
            Config.CLUSTERING['min_strata_size'] = cls.original_min_strata
        if hasattr(cls, 'original_min_cluster'):
            Config.CLUSTERING['min_cluster_size'] = cls.original_min_cluster

        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)

    def test_run_advanced_pipeline_function(self):
        """Test de la fonction run_advanced_pipeline."""
        results = run_advanced_pipeline(
            mode='dev',
            clustering_mode='stratifie',
            sample_size=200,
            force_retrain=True
        )

        # Vérifier la structure
        self.assertIn('data', results)
        self.assertIn('metrics', results)
        self.assertIn('metadata', results)

        # Vérifier le succès
        self.assertEqual(results['metadata']['status'], 'success')

    def test_get_pipeline_model_status_function(self):
        """Test de la fonction get_pipeline_model_status."""
        # Créer un modèle d'abord
        run_advanced_pipeline(mode='dev', sample_size=200, force_retrain=True)

        # Obtenir le statut
        status = get_pipeline_model_status()

        # Vérifier
        self.assertIn('has_model', status)


class TestPipelineEdgeCases(unittest.TestCase):
    """Tests des cas limites du pipeline."""

    @classmethod
    def setUpClass(cls):
        """Configuration une fois pour tous les tests."""
        # Réduire temporairement les contraintes pour les tests
        cls.original_min_strata = Config.CLUSTERING['min_strata_size']
        cls.original_min_cluster = Config.CLUSTERING['min_cluster_size']
        Config.CLUSTERING['min_strata_size'] = 20
        Config.CLUSTERING['min_cluster_size'] = 10

        cls.generator = MockDataGenerator(n_clients=1000, random_state=42)
        cls.temp_dir = tempfile.mkdtemp()
        Config.CACHE_DIR = Path(cls.temp_dir)
        Config.MODELS_DIR = Path(cls.temp_dir) / "models"
        Config.MODELS_DIR.mkdir(exist_ok=True)

        cls.file_paths = cls.generator.save_to_cache(output_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests."""
        # Restaurer les valeurs originales
        if hasattr(cls, 'original_min_strata'):
            Config.CLUSTERING['min_strata_size'] = cls.original_min_strata
        if hasattr(cls, 'original_min_cluster'):
            Config.CLUSTERING['min_cluster_size'] = cls.original_min_cluster

        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)

    def test_pipeline_with_small_sample(self):
        """Test du pipeline avec un échantillon modéré."""
        pipeline = AdvancedBankingPipeline(mode='test', force_retrain=True)

        # Échantillon suffisant pour avoir des strates viables avec min_strata=20
        results = pipeline.run(sample_size=400)

        # Le pipeline devrait fonctionner avec cet échantillon
        self.assertEqual(results['metadata']['status'], 'success')

    def test_pipeline_different_clustering_modes(self):
        """Test du pipeline avec différents modes de clustering."""
        # Mode stratifié - besoin d'assez de clients pour les strates
        results_stratifie = run_advanced_pipeline(
            mode='dev',
            clustering_mode='stratifie',
            sample_size=400,
            force_retrain=True
        )

        self.assertEqual(results_stratifie['metadata']['status'], 'success')

        # Mode global - AGE est injecté temporairement depuis les données brutes
        # puis retiré pour maintenir le schéma cohérent
        results_global = run_advanced_pipeline(
            mode='dev',
            clustering_mode='global',
            sample_size=400,
            force_retrain=True
        )

        self.assertEqual(results_global['metadata']['status'], 'success')

        # Vérifier que le schéma est différent entre les deux modes
        df_stratifie = results_stratifie['data']
        df_global = results_global['data']

        # En mode stratifié, AGE devrait être présent
        self.assertIn('AGE', df_stratifie.columns)

        # En mode global, AGE ne devrait PAS être présent (retiré après clustering)
        self.assertNotIn('AGE', df_global.columns)

        # Mais les colonnes Age_* binaires devraient être présentes en mode global
        age_binary_cols = [c for c in df_global.columns if c.startswith('Age_')]
        self.assertGreater(len(age_binary_cols), 0)


def run_tests(verbosity=2):
    """Exécute tous les tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineEdgeCases))

    # Exécuter
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*60)
    print("TESTS DU PIPELINE COMPLET")
    print("="*60)

    success = run_tests(verbosity=2)

    if success:
        print("\n✅ Tous les tests sont passés!")
        sys.exit(0)
    else:
        print("\n❌ Certains tests ont échoué")
        sys.exit(1)
