#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_runner.py - Script principal pour exécuter tous les tests du projet
"""

import sys
import os
from pathlib import Path
import unittest
import argparse
from datetime import datetime

# Ajout path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import des suites de tests
from test_mock_generator import TestMockDataGenerator, TestMockDataQuality
from test_pipeline import TestAdvancedPipeline, TestPipelineFunctions, TestPipelineEdgeCases
from test_modules import (TestAgeClusterer, TestClientScoringEngine,
                         TestDimensionOptimizer, TestDataExtractor)


class TestRunner:
    """Gestionnaire d'exécution des tests."""

    def __init__(self, verbosity=2):
        """
        Initialise le runner de tests.

        Args:
            verbosity: Niveau de détail (0, 1, 2)
        """
        self.verbosity = verbosity
        self.loader = unittest.TestLoader()

    def get_all_test_suites(self):
        """
        Retourne toutes les suites de tests.

        Returns:
            Dict avec les différentes suites de tests
        """
        return {
            'mock_generator': [TestMockDataGenerator, TestMockDataQuality],
            'pipeline': [TestAdvancedPipeline, TestPipelineFunctions, TestPipelineEdgeCases],
            'modules': [TestAgeClusterer, TestClientScoringEngine,
                       TestDimensionOptimizer, TestDataExtractor]
        }

    def create_suite(self, test_categories=None):
        """
        Crée une suite de tests.

        Args:
            test_categories: Liste des catégories à tester (None = toutes)

        Returns:
            Suite de tests
        """
        suite = unittest.TestSuite()
        all_suites = self.get_all_test_suites()

        if test_categories is None:
            # Toutes les catégories
            test_categories = list(all_suites.keys())

        for category in test_categories:
            if category in all_suites:
                for test_class in all_suites[category]:
                    suite.addTests(self.loader.loadTestsFromTestCase(test_class))
            else:
                print(f"⚠️  Catégorie inconnue: {category}")

        return suite

    def run_tests(self, test_categories=None):
        """
        Exécute les tests.

        Args:
            test_categories: Liste des catégories à tester (None = toutes)

        Returns:
            Tuple (success: bool, result: TestResult)
        """
        print("="*70)
        print("🧪 EXÉCUTION DES TESTS - ML Banking Segmentation")
        print("="*70)

        # Créer la suite
        suite = self.create_suite(test_categories)

        # Afficher les catégories testées
        if test_categories:
            print(f"\n📋 Catégories sélectionnées: {', '.join(test_categories)}")
        else:
            print(f"\n📋 Toutes les catégories de tests")

        print(f"🔢 Nombre total de tests: {suite.countTestCases()}")
        print(f"⏱️  Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Exécuter
        runner = unittest.TextTestRunner(verbosity=self.verbosity)
        result = runner.run(suite)

        # Résumé
        print("\n" + "="*70)
        print("📊 RÉSUMÉ DES TESTS")
        print("="*70)

        print(f"✅ Tests réussis: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"❌ Tests échoués: {len(result.failures)}")
        print(f"💥 Erreurs: {len(result.errors)}")
        print(f"⏭️  Tests ignorés: {len(result.skipped)}")

        # Temps d'exécution
        if hasattr(result, 'stop_time') and hasattr(result, 'start_time'):
            duration = result.stop_time - result.start_time
            print(f"⏱️  Durée totale: {duration:.2f}s")

        success = result.wasSuccessful()

        if success:
            print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
        else:
            print("\n⚠️  CERTAINS TESTS ONT ÉCHOUÉ")

            # Afficher les détails des échecs
            if result.failures:
                print("\n❌ ÉCHECS:")
                for test, traceback in result.failures:
                    print(f"\n  - {test}")
                    print(f"    {traceback.split(chr(10))[0]}")

            if result.errors:
                print("\n💥 ERREURS:")
                for test, traceback in result.errors:
                    print(f"\n  - {test}")
                    print(f"    {traceback.split(chr(10))[0]}")

        print("="*70)

        return success, result

    def run_quick_tests(self):
        """
        Exécute seulement les tests rapides (mock generator).

        Returns:
            Tuple (success: bool, result: TestResult)
        """
        print("⚡ Mode rapide: Tests du générateur de données mock uniquement\n")
        return self.run_tests(test_categories=['mock_generator'])

    def run_integration_tests(self):
        """
        Exécute les tests d'intégration (pipeline complet).

        Returns:
            Tuple (success: bool, result: TestResult)
        """
        print("🔗 Tests d'intégration: Pipeline complet\n")
        return self.run_tests(test_categories=['pipeline'])

    def run_unit_tests(self):
        """
        Exécute les tests unitaires (modules individuels).

        Returns:
            Tuple (success: bool, result: TestResult)
        """
        print("🧩 Tests unitaires: Modules individuels\n")
        return self.run_tests(test_categories=['modules'])


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Runner de tests pour ML Banking Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python test_runner.py                  # Tous les tests
  python test_runner.py --quick          # Tests rapides uniquement
  python test_runner.py --integration    # Tests d'intégration uniquement
  python test_runner.py --unit           # Tests unitaires uniquement
  python test_runner.py --categories mock_generator pipeline
  python test_runner.py -v 1             # Moins de détails
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Exécuter uniquement les tests rapides (mock generator)'
    )

    parser.add_argument(
        '--integration',
        action='store_true',
        help='Exécuter uniquement les tests d\'intégration (pipeline)'
    )

    parser.add_argument(
        '--unit',
        action='store_true',
        help='Exécuter uniquement les tests unitaires (modules)'
    )

    parser.add_argument(
        '--categories',
        nargs='+',
        choices=['mock_generator', 'pipeline', 'modules'],
        help='Catégories de tests spécifiques à exécuter'
    )

    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Niveau de détail (0=minimal, 1=normal, 2=détaillé)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='Liste toutes les catégories de tests disponibles'
    )

    args = parser.parse_args()

    # Créer le runner
    runner = TestRunner(verbosity=args.verbosity)

    # Liste des catégories
    if args.list:
        print("📋 Catégories de tests disponibles:\n")
        all_suites = runner.get_all_test_suites()

        for category, test_classes in all_suites.items():
            print(f"  • {category}:")
            for test_class in test_classes:
                print(f"    - {test_class.__name__}")

        print("\nUtilisation:")
        print("  python test_runner.py --categories <category1> <category2> ...\n")
        return 0

    # Exécuter les tests
    if args.quick:
        success, _ = runner.run_quick_tests()

    elif args.integration:
        success, _ = runner.run_integration_tests()

    elif args.unit:
        success, _ = runner.run_unit_tests()

    elif args.categories:
        success, _ = runner.run_tests(test_categories=args.categories)

    else:
        # Tous les tests par défaut
        success, _ = runner.run_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
