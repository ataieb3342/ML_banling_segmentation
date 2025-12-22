#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_mock_generator.py - Tests pour le générateur de données mock
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

from src.mock_data_generator import MockDataGenerator, create_mock_data
from src.config import Config


class TestMockDataGenerator(unittest.TestCase):
    """Tests du générateur de données mock."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.n_clients = 1000
        self.generator = MockDataGenerator(n_clients=self.n_clients, random_state=42)

    def test_initialization(self):
        """Test de l'initialisation du générateur."""
        self.assertEqual(self.generator.n_clients, self.n_clients)
        self.assertEqual(self.generator.random_state, 42)
        self.assertIsNotNone(self.generator.logger)

    def test_generate_core_data(self):
        """Test de la génération des données core."""
        client_ids = [f"CLI_{i:08d}" for i in range(1, self.n_clients + 1)]
        df_core = self.generator._generate_core_data(client_ids)

        # Vérifier la forme
        self.assertEqual(len(df_core), self.n_clients)
        self.assertGreater(len(df_core.columns), 5)

        # Vérifier les colonnes essentielles
        required_cols = [Config.EXTRACTION['id_column'], 'AGE', 'SEXE',
                        'ANCIENNETE_CLI', 'NB_PRODUITS']
        for col in required_cols:
            self.assertIn(col, df_core.columns)

        # Vérifier les contraintes métier
        self.assertTrue((df_core['AGE'] >= 18).all())
        self.assertTrue((df_core['AGE'] <= 100).all())
        self.assertTrue((df_core['ANCIENNETE_CLI'] >= 0).all())
        self.assertTrue((df_core['NB_PRODUITS'] >= 1).all())
        self.assertTrue(df_core['SEXE'].isin(['M', 'F']).all())

    def test_generate_organizational_data(self):
        """Test de la génération des données organisationnelles."""
        client_ids = [f"CLI_{i:08d}" for i in range(1, self.n_clients + 1)]
        df_org = self.generator._generate_organizational_data(client_ids)

        # Vérifier la forme
        self.assertEqual(len(df_org), self.n_clients)

        # Vérifier les colonnes
        required_cols = [Config.EXTRACTION['id_column'], 'CODE_AGENCE',
                        'REGION', 'CODE_RESEAU']
        for col in required_cols:
            self.assertIn(col, df_org.columns)

        # Vérifier qu'il y a plusieurs agences
        self.assertGreater(df_org['CODE_AGENCE'].nunique(), 1)
        self.assertGreater(df_org['REGION'].nunique(), 1)

    def test_generate_financial_data(self):
        """Test de la génération des données financières."""
        client_ids = [f"CLI_{i:08d}" for i in range(1, self.n_clients + 1)]

        # Générer d'abord les données core (nécessaires pour corrélations)
        df_core = self.generator._generate_core_data(client_ids)
        df_financial = self.generator._generate_financial_data(client_ids, df_core)

        # Vérifier la forme
        self.assertEqual(len(df_financial), self.n_clients)

        # Vérifier les colonnes PNB
        pnb_cols = Config.SCORING['pnb_columns']
        for col in pnb_cols:
            self.assertIn(col, df_financial.columns)

        # Vérifier que le PNB total correspond
        calculated_total = (df_financial[pnb_cols].sum(axis=1)).round(2)
        self.assertTrue(np.allclose(calculated_total, df_financial['PNB_TOTAL'], rtol=0.01))

        # Vérifier les scores
        self.assertTrue((df_financial['CD_NOTE_MIRE'] >= 0).all())
        self.assertTrue((df_financial['CD_NOTE_MIRE'] <= 100).all())
        self.assertTrue((df_financial['CD_NOTE_UNVRS_CRED'] >= 0).all())

    def test_generate_all_data(self):
        """Test de la génération de toutes les données."""
        df_core, df_org, df_financial = self.generator.generate_all_data()

        # Vérifier que toutes les tables ont le même nombre de lignes
        self.assertEqual(len(df_core), self.n_clients)
        self.assertEqual(len(df_org), self.n_clients)
        self.assertEqual(len(df_financial), self.n_clients)

        # Vérifier que les IDs sont cohérents
        self.assertTrue(df_core[Config.EXTRACTION['id_column']].equals(
                       df_org[Config.EXTRACTION['id_column']]))
        self.assertTrue(df_core[Config.EXTRACTION['id_column']].equals(
                       df_financial[Config.EXTRACTION['id_column']]))

    def test_generate_merged_data(self):
        """Test de la génération et fusion des données."""
        df = self.generator.generate_merged_data()

        # Vérifier la forme
        self.assertEqual(len(df), self.n_clients)
        self.assertGreater(len(df.columns), 15)

        # Vérifier qu'on a bien toutes les colonnes importantes
        expected_cols = [
            Config.EXTRACTION['id_column'],
            'AGE', 'SEXE', 'ANCIENNETE_CLI',
            'CODE_AGENCE', 'REGION',
            'PNB_COLL', 'PNB_CRED', 'PNB_SERV', 'PNB_ASSU',
            'CD_NOTE_MIRE'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Pas de valeurs nulles sur les colonnes clés
        self.assertEqual(df[Config.EXTRACTION['id_column']].isna().sum(), 0)
        self.assertEqual(df['AGE'].isna().sum(), 0)

    def test_reproducibility(self):
        """Test de la reproductibilité avec random_state."""
        # Générer deux fois avec le même seed
        gen1 = MockDataGenerator(n_clients=100, random_state=42)
        gen2 = MockDataGenerator(n_clients=100, random_state=42)

        df1 = gen1.generate_merged_data()
        df2 = gen2.generate_merged_data()

        # Les données doivent être identiques
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        """Test que des seeds différents produisent des données différentes."""
        gen1 = MockDataGenerator(n_clients=100, random_state=42)
        gen2 = MockDataGenerator(n_clients=100, random_state=123)

        df1 = gen1.generate_merged_data()
        df2 = gen2.generate_merged_data()

        # Les données doivent être différentes
        self.assertFalse(df1['AGE'].equals(df2['AGE']))

    def test_save_to_cache(self):
        """Test de la sauvegarde dans le cache."""
        # Utiliser un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = self.generator.save_to_cache(output_dir=temp_dir)

            # Vérifier que les fichiers ont été créés
            self.assertIn('core_data', file_paths)
            self.assertIn('organizational_data', file_paths)
            self.assertIn('financial_data', file_paths)

            for path in file_paths.values():
                self.assertTrue(Path(path).exists())

            # Vérifier qu'on peut recharger les données
            import pickle
            for cache_type, path in file_paths.items():
                with open(path, 'rb') as f:
                    cache_obj = pickle.load(f)

                self.assertIn('period', cache_obj)
                self.assertIn('data', cache_obj)
                self.assertEqual(len(cache_obj['data']), self.n_clients)

    def test_get_statistics(self):
        """Test du calcul des statistiques."""
        stats = self.generator.get_statistics()

        # Vérifier les clés
        self.assertIn('n_clients', stats)
        self.assertIn('age', stats)
        self.assertIn('pnb', stats)
        self.assertIn('distribution_sexe', stats)

        # Vérifier les valeurs
        self.assertEqual(stats['n_clients'], self.n_clients)
        self.assertGreater(stats['age']['mean'], 0)
        self.assertGreater(stats['age']['std'], 0)

    def test_create_mock_data_function(self):
        """Test de la fonction utilitaire create_mock_data."""
        # Test de la fonction sans sauvegarde pour éviter les problèmes de cache
        df = create_mock_data(n_clients=500, save_to_cache=False, random_state=42)

        # Vérifier le DataFrame
        self.assertEqual(len(df), 500)
        self.assertGreater(len(df.columns), 15)

        # Vérifier les colonnes essentielles
        expected_cols = [Config.EXTRACTION['id_column'], 'AGE', 'SEXE', 'PNB_TOTAL']
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_correlations(self):
        """Test des corrélations dans les données générées."""
        df = self.generator.generate_merged_data()

        # L'ancienneté devrait être corrélée positivement avec l'âge
        corr_age_anciennete = df['AGE'].corr(df['ANCIENNETE_CLI'])
        self.assertGreater(corr_age_anciennete, 0.1)

        # Le PNB total devrait être corrélé avec le nombre de produits
        corr_pnb_produits = df['PNB_TOTAL'].corr(df['NB_PRODUITS'])
        self.assertGreater(corr_pnb_produits, 0)


class TestMockDataQuality(unittest.TestCase):
    """Tests de la qualité des données mock."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.generator = MockDataGenerator(n_clients=5000, random_state=42)
        self.df = self.generator.generate_merged_data()

    def test_age_distribution(self):
        """Test de la distribution des âges."""
        mean_age = self.df['AGE'].mean()
        std_age = self.df['AGE'].std()

        # La distribution devrait être proche de la configuration
        self.assertAlmostEqual(mean_age, 45, delta=5)
        self.assertAlmostEqual(std_age, 15, delta=5)

    def test_no_missing_values(self):
        """Test qu'il n'y a pas de valeurs manquantes dans les colonnes clés."""
        key_columns = [
            Config.EXTRACTION['id_column'],
            'AGE', 'SEXE', 'PNB_TOTAL', 'CD_NOTE_MIRE'
        ]

        for col in key_columns:
            missing = self.df[col].isna().sum()
            self.assertEqual(missing, 0, f"Colonne {col} a {missing} valeurs manquantes")

    def test_unique_ids(self):
        """Test que tous les IDs sont uniques."""
        id_col = Config.EXTRACTION['id_column']
        n_unique = self.df[id_col].nunique()
        self.assertEqual(n_unique, len(self.df))

    def test_pnb_distribution(self):
        """Test de la distribution du PNB."""
        # Le PNB moyen devrait être positif
        self.assertGreater(self.df['PNB_TOTAL'].mean(), 0)

        # Il devrait y avoir quelques valeurs négatives (clients coûteux)
        n_negative = (self.df['PNB_TOTAL'] < 0).sum()
        self.assertGreater(n_negative, 0)
        self.assertLess(n_negative, len(self.df) * 0.1)  # Moins de 10%

    def test_regional_distribution(self):
        """Test de la distribution régionale."""
        # Il devrait y avoir plusieurs régions
        n_regions = self.df['REGION'].nunique()
        self.assertGreater(n_regions, 5)

        # IDF devrait être la région la plus représentée
        top_region = self.df['REGION'].value_counts().index[0]
        self.assertEqual(top_region, 'ILE_DE_FRANCE')


def run_tests(verbosity=2):
    """Exécute tous les tests."""
    # Créer la suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestMockDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestMockDataQuality))

    # Exécuter
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*60)
    print("TESTS DU GÉNÉRATEUR DE DONNÉES MOCK")
    print("="*60)

    success = run_tests(verbosity=2)

    if success:
        print("\n✅ Tous les tests sont passés!")
        sys.exit(0)
    else:
        print("\n❌ Certains tests ont échoué")
        sys.exit(1)
