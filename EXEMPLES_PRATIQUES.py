"""
╔══════════════════════════════════════════════════════════════════════════╗
║         EXEMPLES PRATIQUES POUR COMPRENDRE LE CODE ML                   ║
║     Copie-colle ces blocs dans ton notebook et exécute-les un par un   ║
╚══════════════════════════════════════════════════════════════════════════╝

Chaque exemple est AUTO-CONTENU (aucune dépendance d'avant).
Cibles : Comprendre ColumnTransformer, Pipeline, GridSearchCV, métriques.
"""

# =============================================================================
# 📦 IMPORT UNIVERSEL
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("✅ Tous les imports loaded");


# =============================================================================
# EXEMPLE 1️⃣  : COMPRENDRE LE PREPROCESSING
# =============================================================================
"""
Objectif : Voir AVANT/APRÈS le preprocessing
"""

print("\n" + "="*70)
print("EXEMPLE 1️⃣  : PREPROCESSING - AVANT/APRÈS")
print("="*70)

# Crée des données TRES simples
data_brutes = pd.DataFrame({
    'Age': [25, 30, np.nan, 45, 50],                    # Numérique avec NaN
    'Revenu': [50000, 75000, 60000, 100000, 80000],     # Numérique grande échelle
    'Type_Maison': ['Studio', 'Villa', 'Apt', 'Villa', 'Studio']  # Catégorique
})

print("\n📊 DONNÉES BRUTES :")
print(data_brutes)
print(f"   Age : min={data_brutes['Age'].min():.0f}, max={data_brutes['Age'].max():.0f} (ranges: 25-50)")
print(f"   Revenu : min={data_brutes['Revenu'].min():.0f}, max={data_brutes['Revenu'].max():.0f} (ranges: 50k-100k)")
print("   ⚠️  PROBLEM : Revenu sur une echelle BEAUCOUP PLUS GRANDE !")

# Sépare colonnes numériques et catégorielles
X = data_brutes.drop(columns=[])  # On a que des features
num_cols = ['Age', 'Revenu']
cat_cols = ['Type_Maison']

# Crée le preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ]), cat_cols)
    ]
)

# Applique le preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Met en DataFrame pour bien visualiser
feature_names = (
    ['Age_scaled', 'Revenu_scaled'] +
    list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(['Type_Maison']))
)
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)

print("\n✅ DONNÉES APRES PREPROCESSING :")
print(X_preprocessed_df)
print(f"   Age recodee : {X_preprocessed_df['Age_scaled'].values}")
print(f"   Revenu recodee : {X_preprocessed_df['Revenu_scaled'].values}")
print(f"   Type_Maison encodee (OneHotEncoder) : colonnes binaires")
print("   ✅ TOUT EST NORMALISE ! Mismo scale, NaN gérés, texte → nombres")


# =============================================================================
# EXEMPLE 2️⃣  : COMPRENDRE LE PIPELINE
# =============================================================================
"""
Objectif : Voir comment les étapes s'enchaînent
"""

print("\n" + "="*70)
print("EXEMPLE 2️⃣  : PIPELINE - CHAÎNE D'ETAPES")
print("="*70)

# Crée des données simples (sans NaN pour simplifier)
np.random.seed(42)
X_demo = pd.DataFrame({
    'feature1': np.random.randn(20) * 100,
    'feature2': np.random.randn(20) * 1000
})
y_demo = 3 * X_demo['feature1'] + 0.01 * X_demo['feature2'] + np.random.randn(20) * 50

print("\nDonnées initiales (très inégales en scale) :")
print(f"  feature1: μ={X_demo['feature1'].mean():.1f}, σ={X_demo['feature1'].std():.1f}")
print(f"  feature2: μ={X_demo['feature2'].mean():.1f}, σ={X_demo['feature2'].std():.1f}")
print("  ⚠️  feature2 est 10x plus grande !")

# Définit le pipeline
simple_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Entraîne
simple_pipeline.fit(X_demo, y_demo)

# Montre la fusion
print(f"\n✅ Pipeline créé avec 2 étapes :")
print(f"  1. StandardScaler → centre & réduit les données")
print(f"  2. LinearRegression → entraîné sur ces données normalisées")
print(f"\n  Prédictions sur premier échantillon :")
pred = simple_pipeline.predict(X_demo.iloc[:3])
print(f"  Réel : {y_demo.iloc[:3].values}")
print(f"  Prédit : {pred}")

# Compare avec un modèle SANS pipeline
lr_no_scale = LinearRegression()
lr_no_scale.fit(X_demo, y_demo)
pred_no_scale = lr_no_scale.predict(X_demo.iloc[:3])
print(f"\n  Sans scaler : {pred_no_scale}")
print(f"  Avec scaler : {pred}")
print(f"  → Les prédictions sont différentes ! Le scaler aide beaucoup !")


# =============================================================================
# EXEMPLE 3️⃣  : COMPRENDRE LES METRIQUES
# =============================================================================
"""
Objectif : Voir comment RMSE, MAE, R² se comportent
"""

print("\n" + "="*70)
print("EXEMPLE 3️⃣  : METRIQUES - RMSE, MAE, R²")
print("="*70)

# Créer 3 prophètes avec qualitès différentes
y_real = np.array([100, 200, 150, 300, 250])
y_pred_bon = np.array([105, 205, 155, 305, 245])      # Petites erreurs
y_pred_moyen = np.array([110, 210, 160, 310, 240])    # Erreurs moyennes
y_pred_mauvais = np.array([150, 250, 100, 400, 200])  # Grandes erreurs

def affiche_metrics(y_real, y_pred, nom):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    erreurs = np.abs(y_real - y_pred)
    
    print(f"\n{nom}")
    print(f"  Erreurs individuelles : {erreurs}")
    print(f"  RMSE : {rmse:.1f}  (√(moyenne des carrés d'erreur))")
    print(f"  MAE  : {mae:.1f}   (moyenne simple des erreurs)")
    print(f"  R²   : {r2:.3f}  (% de variation expliquée)")

affiche_metrics(y_real, y_pred_bon, "📊 Modèle BON")
affiche_metrics(y_real, y_pred_moyen, "📊 Modèle MOYEN")
affiche_metrics(y_real, y_pred_mauvais, "📊 Modèle MAUVAIS")

print("\n✅ RESUME :")
print("  • RMSE pénalise davantage les grandes erreurs (racine du carré)")
print("  • MAE traite toutes les erreurs de façon égale")
print("  • R² dit quelle fraction de la variance on explique (0 à 1)")


# =============================================================================
# EXEMPLE 4️⃣  : COMPRENDRE LA CROSS-VALIDATION
# =============================================================================
"""
Objectif : Voir comment la CV évalue la stabilité
"""

print("\n" + "="*70)
print("EXEMPLE 4️⃣  : CROSS-VALIDATION - STABILITE DU MODELE")
print("="*70)

# Données simples
X_cv = pd.DataFrame(np.random.randn(50, 3), columns=['f1', 'f2', 'f3'])
y_cv = X_cv['f1'] * 2 + X_cv['f2'] * 0.5 + np.random.randn(50) * 0.5

# Deux modèles : Ridge (stable) et Lasso (instable parfois)
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1, max_iter=10000)

# CV scores
cv_ridge = cross_val_score(ridge, X_cv, y_cv, cv=5, scoring='r2')
cv_lasso = cross_val_score(lasso, X_cv, y_cv, cv=5, scoring='r2')

print(f"\n🔄 Cross-Validation avec cv=5 (5 folds) :")
print(f"\nRidge (stable) :")
print(f"  Scores de chaque fold : {cv_ridge}")
print(f"  Moyenne : {cv_ridge.mean():.3f}")
print(f"  Écart-type : {cv_ridge.std():.3f}  ← PETIT (stable ✅)")

print(f"\nLasso (moins stable ici) :")
print(f"  Scores de chaque fold : {cv_lasso}")
print(f"  Moyenne : {cv_lasso.mean():.3f}")
print(f"  Écart-type : {cv_lasso.std():.3f}  ← PLUS GRAND (moins stable ⚠️)")

print(f"\n✅ INTERPRETATION :")
print(f"  • Ridge : stable (écart-type ~0.01) → confidence élevée")
print(f"  • Lasso : pas mal mais moins stable → à surveiller")


# =============================================================================
# EXEMPLE 5️⃣  : COMPRENDRE GRIDSEARCHCV
# =============================================================================
"""
Objectif : Voir comment la grille teste les paramètres
"""

print("\n" + "="*70)
print("EXEMPLE 5️⃣  : GRIDSEARCHCV - RECHERCHE DES MEILLEURS PARAMETRES")
print("="*70)

# Données
np.random.seed(42)
X_grid = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
y_grid = X_grid['f0'] * 3 - X_grid['f1'] + np.random.randn(100) * 0.5

# Train/test split
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_grid, y_grid, test_size=0.2, random_state=42)

# Pipeline
pipeline_grid = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

# PETITE Grille de paramètres (seulement 2×2=4 combinaisons)
small_param_grid = {
    'model__alpha': [0.1, 10.0],
    # Si on avait aussi la sélection :
    # 'feature_selection__k': [3, 5]
}

print("\n🔍 GridSearchCV avec petite grille :")
print(f"  Paramètres à tester : {small_param_grid}")
print(f"  Nombre de combinaisons : 2 (alpha) × 1 (rien d'autre) = 2")
print(f"  CV folds : 5")
print(f"  → Total d'entraînements : 2 × 5 = 10")

# Lance la recherche
grid_search = GridSearchCV(pipeline_grid, small_param_grid, cv=5, scoring='r2', verbose=0)
grid_search.fit(X_train_g, y_train_g)

# Résultats
results_grid = pd.DataFrame(grid_search.cv_results_)
results_simple = results_grid[['param_model__alpha', 'mean_test_score', 'std_test_score']].drop_duplicates()

print(f"\n✅ RESULTATS :")
print(results_simple)
print(f"\n🏆 MEILLEUR PARAMETRE :")
print(f"  alpha = {grid_search.best_params_['model__alpha']}")
print(f"  Score CV (R²) = {grid_search.best_score_:.3f}")

# Évaluation sur test
y_pred_grid = grid_search.best_estimator_.predict(X_test_g)
r2_test = r2_score(y_test_g, y_pred_grid)
print(f"  Score sur test = {r2_test:.3f}")


# =============================================================================
# EXEMPLE 6️⃣  : COMPRENDRE LES VISUALISATIONS
# =============================================================================
"""
Objectif : Créer les 3 graphiques clés
"""

print("\n" + "="*70)
print("EXEMPLE 6️⃣  : VISUALISATIONS - LES 3 GRAPHIQUES CLES")
print("="*70)

# Utilise grid_search du dernier exemple
y_pred = grid_search.predict(X_test_g)

# Figure 1 : Prédictions vs Réel
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test_g, y_pred, alpha=0.6, edgecolors='k')
ax.plot([y_test_g.min(), y_test_g.max()], [y_test_g.min(), y_test_g.max()], 'r--', lw=2, label='Parfait')
ax.set_xlabel('Prix Réel')
ax.set_ylabel('Prix Prédit')
ax.set_title('Prédictions vs Réel')
ax.legend()
plt.tight_layout()
plt.savefig('predictions_vs_real_EXEMPLE.png', dpi=100)
print("\n✅ Figure 1 : Prédictions vs Réel")
print("  → Points proches de la ligne rouge = bonnes prédictions")
plt.show()

# Figure 2 : Résidus
residus = y_test_g - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Résidus vs Prédictions
axes[0].scatter(y_pred, residus, alpha=0.6)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Prédictions')
axes[0].set_ylabel('Résidus')
axes[0].set_title('Résidus vs Prédictions')

# Histogramme
axes[1].hist(residus, bins=15, edgecolor='black')
axes[1].set_xlabel('Résidus')
axes[1].set_ylabel('Fréquence')
axes[1].set_title('Distribution des Résidus')

plt.tight_layout()
plt.savefig('residuals_EXEMPLE.png', dpi=100)
print("\n✅ Figure 2 : Analyse des Résidus")
print("  → Nuage centré autour de 0 = bon");
print("  → Histogramme en cloche = normal = bon")
plt.show()

print("\n" + "="*70)
print("🎉 FIN DES EXEMPLES - Maintenant tu peux tester dans ton notebook !")
print("="*70)
