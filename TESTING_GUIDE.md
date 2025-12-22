# Guide de Test - ML Banking Segmentation

## Vue d'ensemble

Ce projet dispose d'une suite de tests complète et d'un générateur de données mock pour faciliter le développement et valider le pipeline.

## Structure des Tests

```
test/
├── __init__.py              # Initialisation du module de test
├── test_mock_generator.py   # Tests du générateur de données mock (17 tests)
├── test_pipeline.py         # Tests du pipeline complet (tests d'intégration)
├── test_modules.py          # Tests des modules individuels (unitaires)
└── test_runner.py           # Runner principal pour exécuter tous les tests
```

## Démarrage Rapide

### 1. Générer des données mock

```bash
# Générer 5000 clients et sauvegarder dans le cache
python -m src.mock_data_generator -n 5000 --stats
```

Cela créera les fichiers de cache nécessaires pour les tests.

### 2. Exécuter les tests

```bash
# Tous les tests
python test/test_runner.py

# Tests rapides uniquement (générateur mock)
python test/test_runner.py --quick

# Tests d'intégration (pipeline complet)
python test/test_runner.py --integration

# Tests unitaires (modules individuels)
python test/test_runner.py --unit
```

## Utilisation du Générateur de Données Mock

### En ligne de commande

```bash
# Générer 10000 clients avec statistiques
python -m src.mock_data_generator -n 10000 --stats

# Générer sans sauvegarder dans le cache
python -m src.mock_data_generator -n 1000 --no-cache
```

### En Python

```python
from src.mock_data_generator import create_mock_data, MockDataGenerator

# Méthode rapide
df = create_mock_data(n_clients=5000, save_to_cache=True)

# Méthode avancée avec contrôle total
generator = MockDataGenerator(n_clients=10000, random_state=42)

# Générer les 3 tables séparément
df_core, df_org, df_financial = generator.generate_all_data()

# Générer et fusionner
df_merged = generator.generate_merged_data()

# Sauvegarder dans le cache
file_paths = generator.save_to_cache()

# Obtenir les statistiques
stats = generator.get_statistics()
print(f"Âge moyen: {stats['age']['mean']:.1f}")
print(f"PNB moyen: {stats['pnb']['mean']:.2f}")
```

## Tester le Pipeline avec des Données Mock

```python
from src.advanced_pipeline import run_advanced_pipeline

# Exécution complète avec données mock
results = run_advanced_pipeline(
    mode='dev',
    clustering_mode='stratifie',
    sample_size=1000,
    force_retrain=True
)

# Vérifier les résultats
print(f"Status: {results['metadata']['status']}")
print(f"Clients scorés: {len(results['data'])}")
print(f"Action modèle: {results['metadata'].get('model_action')}")

# Accéder aux données scorées
df_scored = results['data']
print(df_scored[['ID_DWR_CLI_CIAL', 'CLIENT_SCORE', 'SEGMENTATION_FINALE']].head())

# Accéder aux métriques
metrics = results['metrics']
print(f"Silhouette moyenne: {metrics['clustering']['global_metrics']['avg_silhouette_score']:.3f}")
```

## Description des Tests

### test_mock_generator.py

**TestMockDataGenerator** (13 tests):
- `test_initialization`: Initialisation du générateur
- `test_generate_core_data`: Génération données démographiques
- `test_generate_organizational_data`: Génération données organisationnelles
- `test_generate_financial_data`: Génération données financières
- `test_generate_all_data`: Génération de toutes les tables
- `test_generate_merged_data`: Fusion des données
- `test_reproducibility`: Reproductibilité avec random_state
- `test_different_seeds`: Variabilité avec différents seeds
- `test_save_to_cache`: Sauvegarde dans le cache
- `test_get_statistics`: Calcul des statistiques
- `test_create_mock_data_function`: Fonction utilitaire
- `test_correlations`: Corrélations entre variables

**TestMockDataQuality** (5 tests):
- `test_age_distribution`: Distribution des âges conforme
- `test_no_missing_values`: Pas de valeurs manquantes
- `test_unique_ids`: Unicité des identifiants
- `test_pnb_distribution`: Distribution réaliste du PNB
- `test_regional_distribution`: Distribution régionale cohérente

### test_pipeline.py

**TestAdvancedPipeline**:
- Tests de chaque étape du pipeline (extraction, optimisation, clustering, scoring)
- Tests des modes d'exécution (dev, test, prod)
- Tests de la gestion des modèles (force_retrain, réutilisation)
- Tests des métriques générées

**TestPipelineFunctions**:
- Tests des fonctions utilitaires
- Tests de `run_advanced_pipeline()`
- Tests de `get_pipeline_model_status()`

**TestPipelineEdgeCases**:
- Tests avec petits échantillons
- Tests avec différents modes de clustering

### test_modules.py

**TestAgeClusterer**:
- Clustering stratifié par âge
- Sauvegarde/chargement de modèles
- Métriques de qualité (silhouette)

**TestClientScoringEngine**:
- Calcul des scores clients
- Segmentation finale
- Potentiels par univers

**TestDimensionOptimizer**:
- Optimisation des dimensions
- Différents modes de clustering

**TestDataExtractor**:
- Extraction de données
- Gestion du cache
- Échantillonnage

## Runner de Tests

Le runner principal (`test_runner.py`) offre plusieurs options:

```bash
# Aide
python test/test_runner.py --help

# Lister les catégories disponibles
python test/test_runner.py --list

# Exécuter des catégories spécifiques
python test/test_runner.py --categories mock_generator pipeline

# Ajuster la verbosité
python test/test_runner.py -v 0  # Minimal
python test/test_runner.py -v 1  # Normal
python test/test_runner.py -v 2  # Détaillé (défaut)
```

### Exemple de sortie

```
======================================================================
🧪 EXÉCUTION DES TESTS - ML Banking Segmentation
======================================================================

📋 Toutes les catégories de tests
🔢 Nombre total de tests: 45
⏱️  Début: 2025-12-22 17:30:00

test_initialization (test_mock_generator.TestMockDataGenerator.test_initialization) ... ok
test_generate_core_data (test_mock_generator.TestMockDataGenerator.test_generate_core_data) ... ok
...

======================================================================
📊 RÉSUMÉ DES TESTS
======================================================================
✅ Tests réussis: 44
❌ Tests échoués: 0
💥 Erreurs: 0
⏭️  Tests ignorés: 0
⏱️  Durée totale: 12.34s

🎉 TOUS LES TESTS SONT PASSÉS!
======================================================================
```

## Workflow de Développement Recommandé

### 1. Configuration initiale

```bash
# Générer des données mock
python -m src.mock_data_generator -n 10000 --stats

# Vérifier que tout fonctionne
python test/test_runner.py --quick
```

### 2. Développement

```bash
# Pendant le développement, tests rapides fréquents
python test/test_runner.py --quick

# Tester un module spécifique
python test/test_modules.py
```

### 3. Avant commit

```bash
# Tous les tests
python test/test_runner.py

# Si succès, commit autorisé
git add .
git commit -m "Feature: nouvelle fonctionnalité"
```

### 4. Tests d'intégration

```bash
# Test complet du pipeline
python test/test_runner.py --integration
```

## Intégration Continue

Pour automatiser les tests avec git hooks, créez `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "🧪 Exécution des tests avant commit..."

python test/test_runner.py -v 1

if [ $? -eq 0 ]; then
    echo "✅ Tests passés - Commit autorisé"
    exit 0
else
    echo "❌ Tests échoués - Commit bloqué"
    echo "Corrigez les erreurs et réessayez"
    exit 1
fi
```

Rendre le hook exécutable:

```bash
chmod +x .git/hooks/pre-commit
```

## Dépannage

### Problème: "Module not found"

```bash
# Assurez-vous d'être à la racine du projet
cd /path/to/ML_banking_segmentation

# Vérifier la structure
ls -la src/ test/
```

### Problème: "Cache not found"

```bash
# Générer les données mock
python -m src.mock_data_generator -n 5000

# Vérifier le cache
ls -lh cache/
```

### Problème: Tests lents

```bash
# Utiliser des échantillons plus petits
python -m src.mock_data_generator -n 1000

# Tests rapides uniquement
python test/test_runner.py --quick
```

## Métriques de Couverture

Pour générer un rapport de couverture de code:

```bash
# Installer coverage
pip install coverage

# Exécuter avec couverture
coverage run test/test_runner.py

# Rapport console
coverage report

# Rapport HTML
coverage html
open htmlcov/index.html
```

## Benchmarking

Pour mesurer les performances:

```python
import time
from src.advanced_pipeline import run_advanced_pipeline

sizes = [500, 1000, 2000, 5000]

for n in sizes:
    start = time.time()
    results = run_advanced_pipeline(
        mode='test',
        sample_size=n,
        force_retrain=True
    )
    duration = time.time() - start
    throughput = n / duration

    print(f"N={n:,}: {duration:.2f}s ({throughput:.0f} clients/s)")
```

## Bonnes Pratiques

1. **Toujours générer des données mock avant les tests**
2. **Utiliser `random_state` pour la reproductibilité**
3. **Tester avec différentes tailles d'échantillons**
4. **Vérifier les logs après chaque exécution**
5. **Nettoyer le cache en cas de doute**:
   ```python
   from src.data_extractor import DataExtractor
   extractor = DataExtractor()
   extractor.clear_cache()
   ```

## Ressources

- [README.md](README.md) - Documentation complète du projet
- [src/mock_data_generator.py](src/mock_data_generator.py) - Code source du générateur
- [test/test_runner.py](test/test_runner.py) - Code source du runner de tests

## Support

Pour toute question ou problème:
1. Consulter les logs dans `logs/`
2. Vérifier le cache dans `cache/`
3. Réexécuter avec `--stats` pour plus de détails
