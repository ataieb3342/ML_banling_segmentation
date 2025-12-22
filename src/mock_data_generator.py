#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mock_data_generator.py - Générateur de données mock pour tests et développement

Génère des données bancaires réalistes pour tester le pipeline sans accès à la base.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
from .config import Config, get_logger


class MockDataGenerator:
    """
    Générateur de données mock pour le pipeline de segmentation bancaire.

    Génère des données réalistes avec distributions cohérentes pour:
    - Données core (âge, sexe, ancienneté)
    - Données organisationnelles (agence, région)
    - Données financières (PNB, scores)
    """

    def __init__(self, n_clients: int = 10000, random_state: int = 42):
        """
        Initialise le générateur.

        Args:
            n_clients: Nombre de clients à générer
            random_state: Seed pour la reproductibilité
        """
        self.n_clients = n_clients
        self.random_state = random_state
        self.logger = get_logger(__name__)
        self.rng = np.random.RandomState(random_state)  # Générateur dédié

        # Configuration des distributions
        self.config = {
            'age': {'min': 18, 'max': 85, 'mean': 45, 'std': 15},
            'anciennete': {'min': 0, 'max': 50, 'mean': 12, 'std': 10},
            'nb_produits': {'min': 1, 'max': 15, 'mean': 5, 'std': 3},
            'pnb': {'mean': 500, 'std': 300, 'min': -100, 'max': 5000},
            'epargne': {'mean': 25000, 'std': 50000, 'min': 0, 'max': 1000000},
            'note_mire': {'min': 0, 'max': 100, 'mean': 50, 'std': 25}
        }

    def generate_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Génère toutes les tables de données.

        Returns:
            Tuple (df_core, df_organizational, df_financial)
        """
        self.logger.info(f"Génération de {self.n_clients:,} clients mock")

        # Générer les IDs uniques
        client_ids = [f"CLI_{i:08d}" for i in range(1, self.n_clients + 1)]

        # Générer chaque table
        df_core = self._generate_core_data(client_ids)
        df_org = self._generate_organizational_data(client_ids)
        df_financial = self._generate_financial_data(client_ids, df_core)

        self.logger.info("Génération de données mock terminée")

        return df_core, df_org, df_financial

    def _generate_core_data(self, client_ids: list) -> pd.DataFrame:
        """
        Génère les données core (démographiques et comportementales).

        Args:
            client_ids: Liste des identifiants clients

        Returns:
            DataFrame avec données core
        """
        n = len(client_ids)

        # Génération avec distributions réalistes
        ages = self.rng.normal(
            self.config['age']['mean'],
            self.config['age']['std'],
            n
        )
        ages = np.clip(ages, self.config['age']['min'], self.config['age']['max']).astype(int)

        # Ancienneté (corrélée avec l'âge)
        anciennete = self.rng.normal(
            self.config['anciennete']['mean'],
            self.config['anciennete']['std'],
            n
        )
        # Les clients plus âgés ont tendance à être plus anciens
        anciennete = anciennete + (ages - 45) * 0.2
        anciennete = np.clip(anciennete, self.config['anciennete']['min'],
                           self.config['anciennete']['max']).astype(int)

        # Nombre de produits (augmente avec ancienneté)
        nb_produits = self.rng.poisson(self.config['nb_produits']['mean'], n)
        nb_produits = nb_produits + (anciennete / 10).astype(int)
        nb_produits = np.clip(nb_produits, self.config['nb_produits']['min'],
                             self.config['nb_produits']['max'])

        # Sexe
        sexe = self.rng.choice(['M', 'F'], n, p=[0.48, 0.52])

        # Nombre de comptes
        nb_comptes = self.rng.poisson(2.5, n)
        nb_comptes = np.clip(nb_comptes, 1, 10)

        # Nombre de majeurs dans le foyer
        nb_majeur = self.rng.choice([1, 2, 3, 4], n, p=[0.35, 0.50, 0.10, 0.05])

        # Montant épargne (log-normal)
        montant_epargne = self.rng.lognormal(
            mean=np.log(self.config['epargne']['mean']),
            sigma=1.5,
            size=n
        )
        montant_epargne = np.clip(montant_epargne,
                                 self.config['epargne']['min'],
                                 self.config['epargne']['max'])

        # Ratio épargne/revenus estimé
        ratio_epargne = montant_epargne / (montant_epargne.mean() * 2)
        ratio_epargne = np.clip(ratio_epargne, 0, 1)

        df = pd.DataFrame({
            Config.EXTRACTION['id_column']: client_ids,
            'AGE': ages,
            'SEXE': sexe,
            'ANCIENNETE_CLI': anciennete,
            'NB_PRODUITS': nb_produits,
            'NB_COMPTES': nb_comptes,
            'NB_MAJEUR': nb_majeur,
            'MONTANT_EPARGNE': montant_epargne.round(2),
            'RATIO_EPARGNE': ratio_epargne.round(3)
        })

        self.logger.info(f"Données core générées: {df.shape}")
        return df

    def _generate_organizational_data(self, client_ids: list) -> pd.DataFrame:
        """
        Génère les données organisationnelles (agence, région).

        Args:
            client_ids: Liste des identifiants clients

        Returns:
            DataFrame avec données organisationnelles
        """
        n = len(client_ids)

        # Génération de codes agences réalistes
        n_agencies = max(10, n // 500)  # ~500 clients par agence
        agencies = [f"AG_{i:04d}" for i in range(1, n_agencies + 1)]

        # Régions (distribution non uniforme)
        regions = ['ILE_DE_FRANCE', 'AUVERGNE_RHONE_ALPES', 'NOUVELLE_AQUITAINE',
                  'OCCITANIE', 'HAUTS_DE_FRANCE', 'GRAND_EST', 'PROVENCE_ALPES',
                  'PAYS_DE_LA_LOIRE', 'BRETAGNE', 'NORMANDIE', 'BOURGOGNE',
                  'CENTRE_VAL_DE_LOIRE']

        region_weights = [0.20, 0.13, 0.10, 0.09, 0.08, 0.08, 0.08,
                         0.06, 0.06, 0.05, 0.04, 0.03]

        # Assignation aléatoire
        client_agencies = self.rng.choice(agencies, n)
        client_regions = self.rng.choice(regions, n, p=region_weights)

        # Codes réseau (hiérarchie)
        reseaux = ['RES_01', 'RES_02', 'RES_03', 'RES_04', 'RES_05']
        client_reseaux = self.rng.choice(reseaux, n)

        df = pd.DataFrame({
            Config.EXTRACTION['id_column']: client_ids,
            'CODE_AGENCE': client_agencies,
            'REGION': client_regions,
            'CODE_RESEAU': client_reseaux
        })

        self.logger.info(f"Données organisationnelles générées: {df.shape}")
        return df

    def _generate_financial_data(self, client_ids: list, df_core: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les données financières (PNB, scores).

        Args:
            client_ids: Liste des identifiants clients
            df_core: DataFrame des données core (pour corrélations)

        Returns:
            DataFrame avec données financières
        """
        n = len(client_ids)

        # Récupérer l'âge et l'ancienneté pour corrélations
        ages = df_core['AGE'].values
        anciennete = df_core['ANCIENNETE_CLI'].values
        nb_produits = df_core['NB_PRODUITS'].values

        # PNB par univers (corrélé avec ancienneté et nb produits)
        base_pnb = self.rng.lognormal(
            mean=np.log(self.config['pnb']['mean']),
            sigma=1.0,
            size=(n, 4)
        )

        # Ajuster selon profil client
        pnb_multiplier = (1 + anciennete / 50) * (1 + nb_produits / 10)
        pnb_multiplier = pnb_multiplier.reshape(-1, 1)

        pnb_values = base_pnb * pnb_multiplier

        # Clipper les valeurs
        pnb_values = np.clip(pnb_values,
                            self.config['pnb']['min'],
                            self.config['pnb']['max'])

        # Certains clients peuvent avoir des PNB négatifs (coûts)
        # ~5% des clients ont un PNB négatif dans au moins un univers
        negative_mask = self.rng.random((n, 4)) < 0.05
        pnb_values[negative_mask] *= -1

        # Score MIRE (note de risque/potentiel)
        # Corrélé positivement avec le PNB total
        total_pnb = pnb_values.sum(axis=1)
        score_mire = 50 + (total_pnb - total_pnb.mean()) / total_pnb.std() * 15
        score_mire = np.clip(score_mire,
                            self.config['note_mire']['min'],
                            self.config['note_mire']['max']).astype(int)

        # Note univers crédit (0 = pas de crédit, >0 = a du crédit)
        # ~30% des clients n'ont pas de crédit
        note_credit = self.rng.choice([0, 1, 2, 3, 4, 5], n,
                                      p=[0.30, 0.20, 0.20, 0.15, 0.10, 0.05])

        df = pd.DataFrame({
            Config.EXTRACTION['id_column']: client_ids,
            'PNB_COLL': pnb_values[:, 0].round(2),
            'PNB_CRED': pnb_values[:, 1].round(2),
            'PNB_SERV': pnb_values[:, 2].round(2),
            'PNB_ASSU': pnb_values[:, 3].round(2),
            'PNB_TOTAL': pnb_values.sum(axis=1).round(2),
            'CD_NOTE_MIRE': score_mire,
            'CD_NOTE_UNVRS_CRED': note_credit
        })

        self.logger.info(f"Données financières générées: {df.shape}")
        return df

    def generate_merged_data(self) -> pd.DataFrame:
        """
        Génère et fusionne toutes les données en un seul DataFrame.

        Returns:
            DataFrame complet avec toutes les données
        """
        df_core, df_org, df_financial = self.generate_all_data()

        # Fusion
        df = df_core.merge(df_org, on=Config.EXTRACTION['id_column'], how='inner')
        df = df.merge(df_financial, on=Config.EXTRACTION['id_column'], how='inner')

        self.logger.info(f"Données fusionnées: {df.shape}")
        return df

    def save_to_cache(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Génère et sauvegarde les données dans le répertoire de cache.

        Args:
            output_dir: Répertoire de sortie (défaut: cache dir du projet)

        Returns:
            Dict avec les chemins des fichiers créés
        """
        if output_dir is None:
            output_dir = Config.CACHE_DIR
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Générer les données
        df_core, df_org, df_financial = self.generate_all_data()

        # Créer des objets cache avec métadonnées (format attendu par DataExtractor)
        from .data_extractor import get_current_period

        current_period = get_current_period()
        timestamp = datetime.now().isoformat()

        cache_objects = {
            'core_data': {
                'period': current_period,
                'created_at': timestamp,
                'n_rows': len(df_core),
                'data': df_core
            },
            'organizational_data': {
                'period': current_period,
                'created_at': timestamp,
                'n_rows': len(df_org),
                'data': df_org
            },
            'financial_data': {
                'period': current_period,
                'created_at': timestamp,
                'n_rows': len(df_financial),
                'data': df_financial
            }
        }

        # Sauvegarder avec pickle
        import pickle

        file_paths = {}
        for cache_type, cache_obj in cache_objects.items():
            cache_path = Config.get_cache_path(cache_type)

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_obj, f)

            file_paths[cache_type] = str(cache_path)
            self.logger.info(f"Sauvegardé: {cache_path} ({len(cache_obj['data']):,} lignes)")

        return file_paths

    def get_statistics(self) -> Dict[str, any]:
        """
        Génère des statistiques sur les données mock.

        Returns:
            Dict avec statistiques descriptives
        """
        df = self.generate_merged_data()

        stats = {
            'n_clients': len(df),
            'age': {
                'mean': df['AGE'].mean(),
                'std': df['AGE'].std(),
                'min': df['AGE'].min(),
                'max': df['AGE'].max()
            },
            'pnb': {
                'mean': df['PNB_TOTAL'].mean(),
                'std': df['PNB_TOTAL'].std(),
                'min': df['PNB_TOTAL'].min(),
                'max': df['PNB_TOTAL'].max()
            },
            'distribution_sexe': df['SEXE'].value_counts().to_dict(),
            'distribution_regions': df['REGION'].value_counts().head().to_dict(),
            'nb_produits_moyen': df['NB_PRODUITS'].mean()
        }

        return stats


def create_mock_data(n_clients: int = 10000,
                     save_to_cache: bool = True,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Fonction utilitaire pour créer rapidement des données mock.

    Args:
        n_clients: Nombre de clients à générer
        save_to_cache: Si True, sauvegarde dans le cache
        random_state: Seed pour reproductibilité

    Returns:
        DataFrame complet avec données mock

    Examples:
        >>> df = create_mock_data(1000)
        >>> df.head()

        >>> df = create_mock_data(50000, save_to_cache=True)
    """
    generator = MockDataGenerator(n_clients=n_clients, random_state=random_state)

    if save_to_cache:
        file_paths = generator.save_to_cache()
        print(f"✅ Données mock sauvegardées dans le cache:")
        for cache_type, path in file_paths.items():
            print(f"   - {cache_type}: {path}")

    return generator.generate_merged_data()


# Point d'entrée pour génération rapide
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Générateur de données mock")
    parser.add_argument('-n', '--n_clients', type=int, default=10000,
                       help='Nombre de clients à générer (défaut: 10000)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Ne pas sauvegarder dans le cache')
    parser.add_argument('--stats', action='store_true',
                       help='Afficher les statistiques')

    args = parser.parse_args()

    print("=" * 60)
    print("GÉNÉRATEUR DE DONNÉES MOCK BANCAIRES")
    print("=" * 60)

    generator = MockDataGenerator(n_clients=args.n_clients)

    if not args.no_cache:
        file_paths = generator.save_to_cache()
        print("\n✅ Fichiers créés:")
        for cache_type, path in file_paths.items():
            print(f"   {cache_type}: {path}")

    if args.stats:
        print("\n📊 STATISTIQUES:")
        stats = generator.get_statistics()

        print(f"\nClients: {stats['n_clients']:,}")
        print(f"\nÂge: {stats['age']['mean']:.1f} ± {stats['age']['std']:.1f} "
              f"({stats['age']['min']} - {stats['age']['max']})")
        print(f"PNB Total: {stats['pnb']['mean']:.2f} ± {stats['pnb']['std']:.2f} "
              f"({stats['pnb']['min']:.2f} - {stats['pnb']['max']:.2f})")
        print(f"Produits moyen: {stats['nb_produits_moyen']:.1f}")

        print("\nDistribution Sexe:")
        for sexe, count in stats['distribution_sexe'].items():
            print(f"   {sexe}: {count:,} ({count/stats['n_clients']*100:.1f}%)")

        print("\nTop 5 Régions:")
        for region, count in stats['distribution_regions'].items():
            print(f"   {region}: {count:,} ({count/stats['n_clients']*100:.1f}%)")

    print("\n✨ Génération terminée!")
