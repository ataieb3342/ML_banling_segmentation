# ML Banking Segmentation

Pipeline de segmentation client bancaire avec clustering stratifie par age et scoring PNB.

## Architecture

```
ML_banking_segmentation/
├── run.py                    # Point d'entree principal (Oozie/Cron)
├── src/
│   ├── config.py             # Configuration centralisee
│   ├── data_extractor.py     # Extraction donnees avec cache intelligent
│   ├── dimension_optimizer.py # Reduction dimensionnelle
│   ├── age_clustering.py     # Clustering stratifie par age
│   ├── client_scoring.py     # Scoring PNB et segmentation
│   ├── advanced_pipeline.py  # Orchestration du pipeline
│   └── mock_data_generator.py # Generateur de donnees mock pour tests
├── test/                     # Tests unitaires et integration
│   ├── test_mock_generator.py # Tests du generateur mock
│   ├── test_pipeline.py      # Tests du pipeline complet
│   ├── test_modules.py       # Tests des modules individuels
│   └── test_runner.py        # Runner principal de tests
├── models/                   # Modeles persistes (.pkl)
├── cache/                    # Cache donnees (avec validation mois/annee)
└── logs/                     # Logs d'execution
```

## Fonctionnement

### Pipeline complet

```
Extraction → Optimisation → Clustering → Scoring → Persistance
    ↓            ↓              ↓           ↓           ↓
 Cache      Reduction       Strates      PNB      Table Spark
(mois/annee) dimensions    par age    intra-cluster
```

### Execution via Oozie/Cron

Le script `run.py` est concu pour etre execute une fois par mois :

```bash
python run.py
```

**Comportement automatique :**
1. Verifie si le cache correspond au mois/annee courant
2. Si cache valide → reutilise les donnees
3. Si cache invalide/absent → relance l'extraction
4. Remplace les donnees du mois courant dans la table finale (evite les doublons)

### Modes d'execution

```python
from src.advanced_pipeline import run_advanced_pipeline

# Mode production (cron mensuel)
results = run_advanced_pipeline(mode='prod', clustering_mode='stratifie')

# Mode developpement (test local)
results = run_advanced_pipeline(mode='dev', clustering_mode='stratifie')

# Mode test avec echantillon
results = run_advanced_pipeline(mode='test', sample_size=10000)

# Forcer le recalcul des modeles
results = run_advanced_pipeline(mode='prod', force_retrain=True)
```

## Gestion du Cache

### Logique du cache intelligent

Le cache est gere par `CacheManager` dans `data_extractor.py` :

- **Un seul fichier cache par type de donnees** (core, organizational, financial)
- **Validation par mois/annee** : Le cache stocke la periode YYYYMM
- **Reutilisation automatique** : Si le cache correspond au mois courant, il est reutilise
- **Extraction automatique** : Si cache absent, perime ou corrompu → extraction depuis la base

```python
# Structure du cache
{
    'period': '202512',           # YYYYMM
    'created_at': '2025-12-01T10:00:00',
    'n_rows': 150000,
    'data': DataFrame
}
```

### Invalidation manuelle du cache

```python
from src.data_extractor import DataExtractor

extractor = DataExtractor()

# Vider tout le cache
extractor.clear_cache()

# Vider un type specifique
extractor.clear_cache('core_data')
extractor.clear_cache('financial_data')
extractor.clear_cache('organizational_data')
```

## Persistance (Table Spark)

### Logique anti-duplication

Dans `advanced_pipeline.py`, la methode `_persist_results` :

1. Ajoute les colonnes temporelles (DATE_EXECUTION, MOIS_ANNEE, ANNEE, MOIS)
2. Verifie si la table existe
3. **Si table existe** : Supprime les donnees du mois courant avant insertion
4. **Si erreur** : Ecrase la table plutot que d'ajouter (evite les doublons)

### Table de sortie

```
{BaseDonnees}.segmentation_clients_scoring
```

**Colonnes cles :**
- ID_DWR_CLI_CIAL : Identifiant client
- CLUSTER_GLOBAL : Cluster assigne (ex: "23-27_C2")
- CLIENT_SCORE : Score PNB normalise
- TOTAL_POTENTIAL : Potentiel de developpement
- SEGMENTATION_FINALE : Label final (A entretenir, A developper, A stimuler, A construire)
- MOIS_ANNEE : Periode de l'execution (YYYYMM)
- DATE_EXECUTION : Timestamp de l'execution

## Intervention Manuelle

### Quand intervenir ?

Les logs signalent automatiquement les alertes. Verifier :
- **Derive demographique** : Age moyen devie de plus de 3 ans
- **Qualite modele faible** : Silhouette < 0.3
- **Couverture clustering faible** : Moins de 80% des clients clusteres

### Actions manuelles

```python
# 1. Forcer le recalcul des modeles de clustering
from src.advanced_pipeline import run_advanced_pipeline
results = run_advanced_pipeline(mode='prod', force_retrain=True)

# 2. Verifier le statut du modele actuel
from src.advanced_pipeline import get_pipeline_model_status
status = get_pipeline_model_status()
print(status)

# 3. Forcer l'extraction (ignorer le cache)
from src.data_extractor import DataExtractor
extractor = DataExtractor(use_cache=False)
df = extractor.extract_all_data()

# 4. Vider le cache et relancer
extractor = DataExtractor()
extractor.clear_cache()
# Puis relancer run.py
```

### Logs a surveiller

Les logs sont dans `logs/` avec le format `{module}_{YYYYMMDD}.log` :

```bash
# Logs du jour
tail -f logs/advanced_pipeline_$(date +%Y%m%d).log

# Historique des executions
cat logs/execution_history.json
```

**Signaux d'alerte dans les logs :**
- `ALERTE DERIVE` : Changement significatif dans les donnees
- `ALERTE QUALITE` : Performances du modele degradees
- `RECOMMANDATION RECALCUL` : Recalcul suggere

## Configuration

### Variables d'environnement

```bash
export DATABASE_NAME="votre_base"  # Base de donnees Spark
```

### Parametres modifiables (src/config.py)

```python
# Extraction
EXTRACTION = {
    'id_column': 'ID_DWR_CLI_CIAL',
    'age_min': 18,
    'age_max': 100,
    'nb_majeur_min': 1
}

# Clustering
CLUSTERING = {
    'age_bins': 5,              # Taille des strates (5 ans)
    'k_range': (2, 6),          # Plage de K pour k-means
    'min_cluster_size': 800,    # Taille minimale d'un cluster
    'min_strata_size': 10000    # Taille minimale d'une strate
}

# Scoring
SCORING = {
    'top_client_pct': 0.25,     # Top 25% pour calcul potentiel
    'pnb_columns': ['PNB_COLL', 'PNB_CRED', 'PNB_SERV', 'PNB_ASSU']
}
```

## Workflow Oozie

### Configuration recommandee

```xml
<workflow-app name="ml_segmentation_mensuelle" xmlns="uri:oozie:workflow:0.5">
    <start to="run_segmentation"/>

    <action name="run_segmentation">
        <shell xmlns="uri:oozie:shell-action:0.3">
            <exec>python</exec>
            <argument>/chemin/vers/run.py</argument>
        </shell>
        <ok to="end"/>
        <error to="fail"/>
    </action>

    <kill name="fail">
        <message>Pipeline segmentation echoue</message>
    </kill>

    <end name="end"/>
</workflow-app>
```

### Coordinator (execution mensuelle)

```xml
<coordinator-app name="coord_segmentation" frequency="${coord:months(1)}"
                 start="2025-01-01T00:00Z" end="2030-12-31T23:59Z">
    <action>
        <workflow>
            <app-path>/chemin/vers/workflow.xml</app-path>
        </workflow>
    </action>
</coordinator-app>
```

## Troubleshooting

| Probleme | Cause probable | Solution |
|----------|---------------|----------|
| Donnees dupliquees | Cache ancien format | `extractor.clear_cache()` puis relancer |
| Modele non trouve | Premier lancement | Normal, le modele sera cree |
| Spark non disponible | Environnement local | Utiliser `mode='dev'` |
| Silhouette faible | Donnees changees | Forcer `force_retrain=True` |
| Cache non reutilise | Mois different | Normal, extraction necessaire |

## Reprise du projet

### Fichiers cles a comprendre

1. **run.py** : Point d'entree, appelle le pipeline
2. **src/advanced_pipeline.py** : Orchestration des etapes
3. **src/data_extractor.py** : Extraction avec cache intelligent
4. **src/age_clustering.py** : Logique de clustering
5. **src/client_scoring.py** : Calcul des scores et segmentation

### Tests locaux

```python
# Test rapide en mode dev
from src.advanced_pipeline import run_advanced_pipeline

results = run_advanced_pipeline(
    mode='dev',
    clustering_mode='stratifie',
    sample_size=5000
)

print(f"Status: {results['metadata']['status']}")
print(f"Clients scores: {len(results['data'])}")
```

### Verification du cache

```python
from src.data_extractor import DataExtractor, get_current_period

extractor = DataExtractor()
print(f"Periode courante: {get_current_period()}")

# Le cache sera automatiquement valide ou invalide selon la periode
df = extractor.extract_all_data()
```

## Specifications Techniques

### Gestion Intelligente des Modeles (Model Lifecycle)

Le pipeline implemente une strategie de **conservation des modeles** pour eviter les recalculs inutiles :

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOGIQUE DE DECISION MODELE                    │
├─────────────────────────────────────────────────────────────────┤
│  Modele existant ?                                               │
│       │                                                          │
│       ├── NON → Entrainement obligatoire (fit_transform)        │
│       │                                                          │
│       └── OUI → force_retrain ?                                  │
│                    │                                             │
│                    ├── OUI → Recalcul force                      │
│                    │                                             │
│                    └── NON → Reutilisation modele (transform)   │
│                              + Log metriques de monitoring       │
└─────────────────────────────────────────────────────────────────┘
```

**Le modele n'est JAMAIS recalcule automatiquement** meme en cas de derive detectee.
Les alertes sont loguees pour decision humaine.

### Detection de Derive (Drift Detection)

A chaque execution, le pipeline compare les nouvelles donnees avec les metadonnees d'entrainement :

| Metrique | Seuil d'alerte | Description |
|----------|----------------|-------------|
| Derive age moyen | > 3 ans | Ecart entre age moyen actuel vs entrainement |
| Derive ecart-type age | > 1 an | Changement dans la dispersion des ages |
| Ratio volume | > 2x ou < 0.5x | Changement significatif du nombre de clients |
| Silhouette | < 0.3 | Qualite du clustering degradee |
| Couverture clustering | < 80% | Trop de clients non-clusteres |
| Strates cluster unique | > 40% | Trop de strates sans segmentation |

**Exemple de log de monitoring :**
```
⚖️  COMPARAISON MODÈLE vs NOUVELLES DONNÉES:
  Dérive âge moyen: 1.2 ans
  Dérive écart-type âge: 0.3
  Ratio volume: 1.05x (152,000 vs 145,000)

🚨 SIGNAUX POUR DÉCISION MANUELLE:
  ✅ AUCUNE ALERTE - Modèle semble adapté aux nouvelles données
```

### Algorithme de Scoring PNB

Le scoring utilise une methodologie en 5 etapes :

```
1. PONDERATION RELATIVE
   ┌──────────────────────────────────────────────────────────┐
   │ Pour chaque cluster :                                    │
   │   poids_PNB_i = |moyenne_PNB_i| / Σ|moyenne_PNB_j|       │
   │                                                          │
   │ Exemple cluster "33-37_C2" :                             │
   │   PNB_COLL: 45% | PNB_CRED: 30% | PNB_SERV: 15% | ...    │
   └──────────────────────────────────────────────────────────┘

2. NORMALISATION MIN-MAX (par cluster)
   ┌──────────────────────────────────────────────────────────┐
   │ PNB_NORM = (PNB - min_cluster) / (max_cluster - min)     │
   │                                                          │
   │ Resultat : valeur entre 0 et 1 pour chaque univers       │
   └──────────────────────────────────────────────────────────┘

3. SCORE CLIENT PONDERE
   ┌──────────────────────────────────────────────────────────┐
   │ CLIENT_SCORE = Σ (PNB_NORM_i × poids_i)                  │
   │                                                          │
   │ Score final entre 0 et 1                                 │
   └──────────────────────────────────────────────────────────┘

4. CALCUL DES POTENTIELS (Top 25%)
   ┌──────────────────────────────────────────────────────────┐
   │ Pour chaque cluster :                                    │
   │   1. Identifier les 25% meilleurs clients (par score)    │
   │   2. Calculer mediane PNB de ces top clients             │
   │   3. Potentiel = max(0, mediane_top - PNB_client)        │
   │                                                          │
   │ Cas special PNB_COLL : basé sur ratio epargne            │
   │ Cas special PNB_CRED : potentiel si CD_NOTE_UNVRS_CRED=0 │
   └──────────────────────────────────────────────────────────┘

5. SEGMENTATION FINALE
   ┌──────────────────────────────────────────────────────────┐
   │ SEGMENT_PNB (1-4) basé sur quartiles intra-cluster       │
   │   Q75+ → 4 | Q50-Q75 → 3 | Q25-Q50 → 2 | <Q25 → 1        │
   │                                                          │
   │ SEGMENT_MIRE (1-4) basé sur CD_NOTE_MIRE                 │
   │   75+ → 4 | 50-74 → 3 | 25-49 → 2 | <25 → 1              │
   │                                                          │
   │ SEGMENT_FINAL = SEGMENT_PNB + SEGMENT_MIRE (2-8)         │
   │                                                          │
   │ Labels :                                                 │
   │   8      → "A entretenir"  (clients premium)             │
   │   6-7    → "A developper"  (fort potentiel)              │
   │   4-5    → "A stimuler"    (potentiel moyen)             │
   │   2-3    → "A construire"  (relation a batir)            │
   └──────────────────────────────────────────────────────────┘
```

### Clustering Stratifie par Age

**Architecture du clustering :**

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLUSTERING PAR STRATES                       │
├─────────────────────────────────────────────────────────────────┤
│  Strate    │  Ages   │ K-means  │  Contraintes                  │
│────────────┼─────────┼──────────┼───────────────────────────────│
│  18-22     │ 18-22   │ K=2-6    │ min_cluster=800               │
│  23-27     │ 23-27   │ K=2-6    │ min_strata=10,000             │
│  28-32     │ 28-32   │ K=2-6    │                               │
│  ...       │ ...     │ ...      │                               │
│  58-62     │ 58-62   │ K=2-6    │                               │
│  63+       │ 63+     │ K=2-6    │                               │
└─────────────────────────────────────────────────────────────────┘

Optimisation K :
  1. Pour K in [2, min(6, n_clients/800)] :
     - Entrainer K-means
     - Verifier taille min cluster >= 800
     - Calculer silhouette score
  2. Selectionner K avec meilleur silhouette
  3. Fallback K=1 si aucun K valide (cluster unique)
```

**Evaluation qualite (Silhouette Score) :**

| Score | Niveau | Action |
|-------|--------|--------|
| > 0.7 | Excellent | Aucune action |
| 0.5 - 0.7 | Bon | Surveillance normale |
| 0.3 - 0.5 | Acceptable | Surveillance accrue |
| < 0.3 | Faible | Recalcul recommande |

### Serialisation des Modeles

**Structure du fichier pickle (.pkl) :**

```python
{
    'model_version': '20251222_143052',      # Timestamp creation
    'age_strata': {                           # Definition des strates
        '18-22': (18, 22),
        '23-27': (23, 27),
        ...
    },
    'scaler_models': {                        # StandardScaler par strate
        '18-22': StandardScaler(),
        '23-27': StandardScaler(),
        ...
    },
    'kmeans_models': {                        # KMeans par strate (si K>1)
        '23-27': KMeans(n_clusters=3),
        '33-37': KMeans(n_clusters=4),
        ...
    },
    'quality_metrics': {
        'global_metrics': {
            'avg_silhouette_score': 0.52,
            'total_clusters': 28,
            'viable_strata': 10,
            ...
        },
        'strata_details': { ... }
    },
    'training_metadata': {
        'training_date': '2025-12-01T10:00:00',
        'n_samples': 145000,
        'n_features': 8,
        'age_mean': 42.3,
        'age_std': 14.2,
        'age_min': 18,
        'age_max': 95
    }
}
```

**Historique des modeles (clustering_history.json) :**

```json
[
    {
        "model_version": "20251101_100000",
        "training_date": "2025-11-01T10:00:00",
        "n_samples": 142000,
        "avg_silhouette_score": 0.51,
        "total_clusters": 27,
        "model_path": "models/age_clustering_20251101_100000.pkl"
    },
    {
        "model_version": "20251201_100000",
        "training_date": "2025-12-01T10:00:00",
        "n_samples": 145000,
        "avg_silhouette_score": 0.52,
        "total_clusters": 28,
        "model_path": "models/age_clustering_20251201_100000.pkl"
    }
]
```

### Metriques de Production

**Metriques cles a monitorer :**

```
┌─────────────────────────────────────────────────────────────────┐
│                    DASHBOARD METRIQUES                          │
├─────────────────────────────────────────────────────────────────┤
│ CLUSTERING                                                      │
│   • Silhouette score moyen           [0.3 ─────●───── 0.7]      │
│   • Couverture clustering            [80% ─────●───── 100%]     │
│   • Strates viables                  [8/10 strates]             │
│                                                                 │
│ SCORING                                                         │
│   • Potentiel total                  [2.5M€]                    │
│   • Score client moyen               [0.45]                     │
│   • Couverture scoring               [95%]                      │
│                                                                 │
│ SEGMENTATION                                                    │
│   • A entretenir    ████████░░░░░░░░  15%                       │
│   • A developper    ██████████████░░  35%                       │
│   • A stimuler      ████████████░░░░  30%                       │
│   • A construire    ████████░░░░░░░░  20%                       │
│                                                                 │
│ DERIVE                                                          │
│   • Age modele                       [45 jours]                 │
│   • Derive demographique             [Faible]                   │
│   • Ratio volume                     [1.02x]                    │
└─────────────────────────────────────────────────────────────────┘
```

**API de verification du statut :**

```python
from src.advanced_pipeline import get_pipeline_model_status

status = get_pipeline_model_status()
# {
#     'has_model': True,
#     'model_path': 'models/age_clustering_20251201_100000.pkl',
#     'model_version': '20251201_100000',
#     'training_date': '2025-12-01T10:00:00',
#     'n_samples': 145000,
#     'avg_silhouette': 0.52,
#     'total_clusters': 28,
#     'viable_strata': 10
# }
```

### Schema de Donnees

**Table de sortie (segmentation_clients_scoring) :**

| Colonne | Type | Description |
|---------|------|-------------|
| ID_DWR_CLI_CIAL | STRING | Identifiant client unique |
| AGE | INT | Age du client |
| CLUSTER_STRATE | STRING | Nom de la strate (ex: "33-37") |
| CLUSTER_LOCAL | INT | Numero cluster dans la strate (0, 1, 2...) |
| CLUSTER_GLOBAL | STRING | Cluster complet (ex: "33-37_C2") |
| CLIENT_SCORE | FLOAT | Score PNB pondere normalise [0-1] |
| TOTAL_POTENTIAL | FLOAT | Potentiel de developpement en euros |
| POTENTIAL_PNB_COLL | FLOAT | Potentiel collecte |
| POTENTIAL_PNB_CRED | FLOAT | Potentiel credit |
| POTENTIAL_PNB_SERV | FLOAT | Potentiel services |
| POTENTIAL_PNB_ASSU | FLOAT | Potentiel assurance |
| SEGMENT_PNB | INT | Segment PNB (1-4) |
| SEGMENT_MIRE | INT | Segment MIRE (1-4) |
| SEGMENT_FINAL | INT | Score combine (2-8) |
| SEGMENTATION_FINALE | STRING | Label final |
| DATE_EXECUTION | TIMESTAMP | Date/heure d'execution |
| MOIS_ANNEE | STRING | Periode (YYYYMM) |
| ANNEE | INT | Annee |
| MOIS | INT | Mois |

## Tests et Developpement

### Generateur de Donnees Mock

Le projet inclut un generateur de donnees mock pour tester le pipeline sans acces a la base de production.

**Generation rapide de donnees:**

```bash
# Generer 10000 clients et sauvegarder dans le cache
python -m src.mock_data_generator -n 10000

# Generer avec statistiques
python -m src.mock_data_generator -n 5000 --stats

# Generer sans sauvegarder
python -m src.mock_data_generator -n 1000 --no-cache
```

**Utilisation en Python:**

```python
from src.mock_data_generator import create_mock_data, MockDataGenerator

# Generation rapide
df = create_mock_data(n_clients=5000, save_to_cache=True)

# Generateur avec controle total
generator = MockDataGenerator(n_clients=10000, random_state=42)
df_core, df_org, df_financial = generator.generate_all_data()

# Sauvegarder dans le cache
file_paths = generator.save_to_cache()

# Obtenir les statistiques
stats = generator.get_statistics()
```

**Caracteristiques des donnees mock:**

- Distributions realistes (age, PNB, anciennete)
- Correlations coherentes (age/anciennete, PNB/nb_produits)
- Format compatible avec le pipeline
- Reproductibilite avec random_state
- Cache automatique avec structure periode

### Execution des Tests

Le projet dispose d'une suite de tests complete pour valider tous les modules.

**Runner de tests principal:**

```bash
# Executer tous les tests
python test/test_runner.py

# Tests rapides uniquement (mock generator)
python test/test_runner.py --quick

# Tests d'integration (pipeline complet)
python test/test_runner.py --integration

# Tests unitaires (modules individuels)
python test/test_runner.py --unit

# Categories specifiques
python test/test_runner.py --categories mock_generator pipeline

# Moins de verbosity
python test/test_runner.py -v 1

# Lister toutes les categories disponibles
python test/test_runner.py --list
```

**Tests individuels:**

```bash
# Tests du generateur mock
python test/test_mock_generator.py

# Tests du pipeline complet
python test/test_pipeline.py

# Tests des modules (clustering, scoring, etc.)
python test/test_modules.py
```

### Organisation des Tests

**test/test_mock_generator.py:**
- TestMockDataGenerator: Tests du generateur
- TestMockDataQuality: Tests de la qualite des donnees

**test/test_pipeline.py:**
- TestAdvancedPipeline: Tests du pipeline complet
- TestPipelineFunctions: Tests des fonctions utilitaires
- TestPipelineEdgeCases: Tests des cas limites

**test/test_modules.py:**
- TestAgeClusterer: Tests du clustering stratifie
- TestClientScoringEngine: Tests du moteur de scoring
- TestDimensionOptimizer: Tests de l'optimisation
- TestDataExtractor: Tests de l'extraction

### Workflow de Developpement Recommande

**1. Generer des donnees mock:**

```bash
python -m src.mock_data_generator -n 10000 --stats
```

**2. Tester le pipeline en mode dev:**

```python
from src.advanced_pipeline import run_advanced_pipeline

# Execution locale avec donnees mock
results = run_advanced_pipeline(
    mode='dev',
    clustering_mode='stratifie',
    sample_size=1000,
    force_retrain=True
)

print(f"Status: {results['metadata']['status']}")
print(f"Clients scores: {len(results['data'])}")
```

**3. Executer les tests:**

```bash
# Tests rapides pendant le developpement
python test/test_runner.py --quick

# Tests complets avant commit
python test/test_runner.py
```

**4. Verifier les resultats:**

```bash
# Consulter les logs
tail -f logs/advanced_pipeline_$(date +%Y%m%d).log

# Verifier les modeles generes
ls -lh models/

# Verifier le cache
ls -lh cache/
```

### Integration Continue

**Script de validation pre-commit:**

```bash
#!/bin/bash
# Executer avant chaque commit

echo "Execution des tests..."
python test/test_runner.py -v 1

if [ $? -eq 0 ]; then
    echo "✅ Tests passes - Commit autorise"
    exit 0
else
    echo "❌ Tests echoues - Commit bloque"
    exit 1
fi
```

### Benchmarking et Performance

**Test de performance du pipeline:**

```python
from src.advanced_pipeline import run_advanced_pipeline
import time

# Tester avec differentes tailles
for n in [1000, 5000, 10000, 50000]:
    start = time.time()
    results = run_advanced_pipeline(
        mode='test',
        sample_size=n,
        force_retrain=True
    )
    duration = time.time() - start

    print(f"N={n:,}: {duration:.2f}s ({n/duration:.0f} clients/s)")
```

## Auteur

Equipe Data Science - Credit Agricole

## Version

4.3 - Decembre 2025
- Cache intelligent base sur mois/annee
- Prevention des doublons lors de la persistance
- Logs ameliores pour le monitoring
- Generateur de donnees mock integre
- Suite de tests complete (unitaires + integration)
- Documentation technique complete
