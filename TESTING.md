# Guide de Tests - ML Banking Segmentation

Ce document décrit la structure des tests, comment les exécuter, et les bonnes pratiques pour maintenir la qualité du code.

## Structure des Tests

```
test/
├── test_modules.py       # Tests unitaires des modules individuels
├── test_pipeline.py      # Tests d'intégration du pipeline complet
├── test_mock_generator.py # Tests du générateur de données mock
└── run_tests.py          # Script pour exécuter tous les tests
```

## Exécution des Tests

### Exécuter tous les tests

```bash
# Avec pytest (recommandé)
python -m pytest test/ -v

# Avec le script personnalisé
python test/run_tests.py
```

### Exécuter un fichier de tests spécifique

```bash
python -m pytest test/test_modules.py -v
python -m pytest test/test_pipeline.py -v
python -m pytest test/test_mock_generator.py -v
```

### Exécuter un test spécifique

```bash
python -m pytest test/test_modules.py::TestAgeClusterer::test_fit_transform -v
```

### Options utiles

```bash
# Afficher les prints et logs
python -m pytest test/ -v -s

# Arrêter au premier échec
python -m pytest test/ -v -x

# Afficher les 5 tests les plus lents
python -m pytest test/ -v --durations=5

# Exécuter avec couverture de code
python -m pytest test/ --cov=src --cov-report=html
```

## Modules Testés

### 1. DataExtractor (`test_modules.py::TestDataExtractor`)

| Test | Description |
|------|-------------|
| `test_extractor_initialization` | Vérifie l'initialisation correcte de l'extracteur |
| `test_get_current_period` | Vérifie le format de la période (AAAAMM) |
| `test_get_sample` | Vérifie l'échantillonnage des données |
| `test_extract_financial_data` | Vérifie l'extraction des données financières |
| `test_extract_all_data` | Vérifie l'extraction complète avec fusion |
| `test_cache_manager` | Vérifie le système de cache |

### 2. DimensionOptimizer (`test_modules.py::TestDimensionOptimizer`)

| Test | Description |
|------|-------------|
| `test_optimizer_initialization` | Vérifie l'initialisation avec différents modes |
| `test_transform` | Vérifie la transformation des données |
| `test_transform_without_ids` | Vérifie le comportement sans colonne ID |
| `test_different_clustering_modes` | Vérifie les modes 'stratifie' et 'global' |

### 3. AgeClusterer (`test_modules.py::TestAgeClusterer`)

| Test | Description |
|------|-------------|
| `test_clusterer_initialization` | Vérifie l'initialisation du clusterer |
| `test_age_strata_creation` | Vérifie la création des strates d'âge |
| `test_fit_transform` | Vérifie le clustering complet |
| `test_clustering_quality_metrics` | Vérifie les métriques de qualité |
| `test_model_save_and_load` | Vérifie la persistance du modèle |
| `test_transform_without_fit` | Vérifie l'erreur si transform sans fit |
| `test_non_clustered_handling` | Vérifie la gestion des clients non clusterisés |

### 4. ClientScoringEngine (`test_modules.py::TestClientScoringEngine`)

| Test | Description |
|------|-------------|
| `test_scoring_engine_initialization` | Vérifie l'initialisation du moteur de scoring |
| `test_calculate_scores` | Vérifie le calcul des scores |
| `test_score_ranges` | Vérifie que les scores sont dans [0, 1] |
| `test_potential_columns` | Vérifie les colonnes de potentiel |
| `test_segmentation_labels` | Vérifie les labels de segmentation |
| `test_segment_distribution` | Vérifie la distribution des segments |
| `test_scoring_metrics` | Vérifie les métriques de scoring |

### 5. Configuration (`test_modules.py::TestConfig`)

| Test | Description |
|------|-------------|
| `test_get_logger` | Vérifie la création de loggers |
| `test_get_logger_same_name` | Vérifie le singleton des loggers |
| `test_config_paths_exist` | Vérifie les chemins de configuration |
| `test_config_clustering_params` | Vérifie les paramètres de clustering |
| `test_config_scoring_params` | Vérifie les paramètres de scoring |
| `test_cache_manager_invalid_path` | Vérifie la gestion des chemins invalides |
| `test_cache_manager_expired_period` | Vérifie la détection de cache expiré |

### 6. Tests de Robustesse (`test_modules.py::TestRobustness`)

| Test | Description |
|------|-------------|
| `test_age_clusterer_empty_dataframe` | Vérifie le comportement avec DataFrame vide |
| `test_age_clusterer_missing_age_column` | Vérifie l'erreur si colonne AGE manquante |
| `test_age_clusterer_with_nan_ages` | Vérifie la gestion des valeurs NaN |
| `test_age_clusterer_all_same_age` | Vérifie le cas où tous les clients ont le même âge |
| `test_dimension_optimizer_empty_dataframe` | Vérifie le comportement avec DataFrame vide |
| `test_dimension_optimizer_missing_columns` | Vérifie la gestion des colonnes manquantes |
| `test_data_extractor_without_cache` | Vérifie le comportement sans cache |
| `test_scoring_engine_empty_cluster_data` | Vérifie le comportement avec données vides |

## Tests du Pipeline (`test_pipeline.py`)

### TestAdvancedPipeline

| Test | Description |
|------|-------------|
| `test_pipeline_initialization` | Initialisation du pipeline |
| `test_pipeline_modes` | Modes dev, test, prod |
| `test_invalid_mode` | Erreur si mode invalide |
| `test_invalid_clustering_mode` | Erreur si mode clustering invalide |
| `test_pipeline_run_dev_mode` | Exécution complète en mode dev |
| `test_pipeline_run_test_mode` | Exécution avec échantillon réduit |
| `test_pipeline_extraction_step` | Étape d'extraction |
| `test_pipeline_optimization_step` | Étape d'optimisation |
| `test_pipeline_clustering_step` | Étape de clustering |
| `test_pipeline_scoring_step` | Étape de scoring |
| `test_pipeline_with_force_retrain` | Ré-entraînement forcé |
| `test_pipeline_without_force_retrain` | Réutilisation du modèle |
| `test_pipeline_metrics` | Génération des métriques |
| `test_get_model_status` | Statut du modèle |

### TestPipelineEdgeCases

| Test | Description |
|------|-------------|
| `test_pipeline_with_small_sample` | Pipeline avec échantillon modéré |
| `test_pipeline_different_clustering_modes` | Différents modes de clustering |

## Tests du Générateur Mock (`test_mock_generator.py`)

### TestMockDataGenerator

| Test | Description |
|------|-------------|
| `test_initialization` | Initialisation du générateur |
| `test_generate_core_data` | Génération des données core |
| `test_generate_financial_data` | Génération des données financières |
| `test_generate_organizational_data` | Génération des données organisationnelles |
| `test_generate_merged_data` | Fusion des données |
| `test_generate_all_data` | Génération complète |
| `test_get_statistics` | Statistiques des données |
| `test_save_to_cache` | Sauvegarde en cache |
| `test_reproducibility` | Reproductibilité avec seed |
| `test_different_seeds` | Différence avec seeds différents |
| `test_correlations` | Corrélations attendues |

### TestMockDataQuality

| Test | Description |
|------|-------------|
| `test_unique_ids` | Unicité des identifiants |
| `test_no_missing_values` | Absence de valeurs manquantes |
| `test_age_distribution` | Distribution des âges [18, 85] |
| `test_pnb_distribution` | Distribution du PNB |
| `test_regional_distribution` | Distribution régionale |

## Isolation des Tests

Les tests utilisent des répertoires temporaires pour éviter les effets de bord :

```python
@classmethod
def setUpClass(cls):
    # Sauvegarder la configuration originale
    cls.original_min_strata = Config.CLUSTERING['min_strata_size']
    cls.original_models_dir = Config.MODELS_DIR

    # Créer un répertoire temporaire
    cls.temp_dir = tempfile.mkdtemp()
    Config.MODELS_DIR = Path(cls.temp_dir)

@classmethod
def tearDownClass(cls):
    # Restaurer la configuration
    Config.CLUSTERING['min_strata_size'] = cls.original_min_strata
    Config.MODELS_DIR = cls.original_models_dir

    # Nettoyer le répertoire temporaire
    shutil.rmtree(cls.temp_dir)
```

## Bonnes Pratiques

### Ajouter un nouveau test

1. Identifier le module à tester
2. Ajouter le test dans la classe appropriée
3. Utiliser `setUp`/`tearDown` pour l'isolation
4. Vérifier les assertions clés

```python
def test_nouvelle_fonctionnalite(self):
    """Description claire du test."""
    # Arrange - Préparer les données
    input_data = ...

    # Act - Exécuter la fonction
    result = function_to_test(input_data)

    # Assert - Vérifier les résultats
    self.assertIsNotNone(result)
    self.assertEqual(result['key'], expected_value)
```

### Conventions de nommage

- `test_<fonctionnalite>` : Test nominal
- `test_<fonctionnalite>_<cas_specifique>` : Test d'un cas particulier
- `test_<fonctionnalite>_empty_data` : Test avec données vides
- `test_<fonctionnalite>_invalid_input` : Test avec entrée invalide

### Tailles d'échantillon recommandées

| Contexte | Taille minimale |
|----------|-----------------|
| Tests unitaires simples | 100-200 |
| Tests de clustering | 400-500 |
| Tests de pipeline complet | 500-1000 |

## Couverture Actuelle

| Module | Couverture |
|--------|------------|
| `data_extractor.py` | Tests d'extraction et cache |
| `dimension_optimizer.py` | Tests de transformation |
| `age_clustering.py` | Tests complets avec persistance |
| `client_scoring.py` | Tests de scoring et segmentation |
| `advanced_pipeline.py` | Tests d'intégration |
| `mock_data_generator.py` | Tests de génération |
| `config.py` | Tests de configuration et logging |

## Limitations Connues

1. **Warnings sklearn** : Certains tests génèrent des `ConvergenceWarning` de sklearn quand le nombre de clusters distinct est inférieur au nombre demandé. C'est attendu avec des petits échantillons.

## Notes Techniques

### Mode 'global' et AgeClusterer

Le mode 'global' dans `DimensionOptimizer` transforme la colonne AGE en colonnes binaires (Age_Jeunes, Age_Actifs, etc.). Cependant, `AgeClusterer` nécessite une colonne AGE numérique pour la stratification.

**Solution implémentée** : Le pipeline (`advanced_pipeline.py`) gère cette incompatibilité en :
1. Détectant le mode 'global' et l'absence de colonne AGE
2. Injectant temporairement la colonne AGE depuis les données brutes
3. Exécutant le clustering normalement
4. Retirant la colonne AGE temporaire du résultat pour maintenir un schéma cohérent

Cela permet d'utiliser les deux modes de clustering tout en conservant des schémas de sortie distincts et prévisibles.

## Exécution CI/CD

Pour intégration continue, utiliser :

```bash
python -m pytest test/ -v --tb=short --junitxml=test-results.xml
```

Cette commande génère un rapport XML compatible avec la plupart des systèmes CI/CD.
