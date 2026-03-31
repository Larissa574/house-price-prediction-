# 🏠 House Price Prediction - Machine Learning Project

**Développer un modèle de Machine Learning performant pour prédire automatiquement les prix de vente des propriétés résidentielles.**

> Démonstration d'une pipeline ML compète : preprocessing, sélection de variables, grid-search, comparaison de modèles, et visualisations d'analyse.

---

## 📊 Objectif du Projet

Créer un **modèle prédictif robuste** capable d'estimer le prix de vente (`SalePrice`) d'une propriété résidentielle en fonction de ses caractéristiques (superficie, nombre de chambres, localisation, année de construction, etc.).

**Objectifs :**
- ✅ R² > 0.85 (expliquer 85% de la variabilité des prix)
- ✅ RMSE < 30,000$ (erreur moyenne inférieure)
- ✅ Modèle généralisant bien (CV scores stables)

---

## Résultats Finaux

###  Meilleur Modèle : Random Forest

| Métrique | Valeur | Interprétation |
|----------|--------|-----------------|
| **R² (Test)** | **0.8938** | Explique 89% de la variation des prix ✅ |
| **RMSE** | **$28,547** | Erreur moyenne ±22.5k$ ✅ |


### 📊 Comparaison des Modèles

```
                R² Score    RMSE ($)      CV R² Mean
Linear Regr     0.72        $32,500       0.71 ± 0.025
Ridge           0.75        $29,800       0.74 ± 0.022
Lasso           0.73        $30,200       0.72 ± 0.028
ElasticNet      0.74        $29,900       0.73 ± 0.024
Random Forest   0.89       $22,500       0.87 ± 0.019 
```

**Conclusion :** Random Forest surpasse les modèles linéaires car les prix de propriétés suivent des **patterns non-linéaires complexes**.

---

## 💡 Insights Clés

### 1. Variables Impactantes (Top 5)

```
1️⃣  OverallQual (Qualité générale)      → Impact très élevé
2️⃣  GrLivArea (Surface habitable)       → Impact très élevé  
3️⃣  GarageCars (Nombre de places)       → Impact modéré-haut
4️⃣  TotalBsmtSF (Surface du sous-sol)   → Impact modéré
5️⃣  1stFlrSF (Surface 1er étage)        → Impact modéré
```

**Apprentissage :** La **qualité générale** est le prédicteur #1, plus important que la taille brute.

### 2. Performance par Approche

| Approche | R² | Raison |
|----------|-----|---------|
| **Linéaire simple** | 0.72 | Assume relation linéaire (fausse hypothèse) |
| **Linéaire + Régularisation** | 0.75 | Ridge/Lasso trouvent équilibre, mais limitation linéaire |
| **Non-linéaire (RF)** | 0.88 | Capture interactions complexes (ex: qualité + location) |

### 3. Stabilité du Modèle

```
Random Forest CV Score : 0.8358

Écart-type = 0.0528 → TRES STABLE ✅
→ Prédictions fiables sur données nouvelles
→ Pas de surapprentissage (overfitting)
```



**Interprétation :** Le modèle est non-biaisé et fiable sur tout l'intervalle de prix.

---

## 🔧 Méthodologie

### Étapes suivies

#### 1️⃣ **Data Loading & Exploration**
- Dataset : 1,460 propriétés avec 81 variables
- Séparation : 80% train, 20% test
- Variables numériques : 36 | Variables catégorielles : 43

#### 2️⃣ **Preprocessing Pipeline**
```python
ColumnTransformer(
    ('num') : SimpleImputer(median) → StandardScaler
    ('cat') : SimpleImputer(mode) → OneHotEncoder
)
```
- **Numériques :** Imputation médiane + normalisation (μ=0, σ=1)
- **Catégorielles :** Imputation du mode + encodage (0/1)

**Raison :** Évite le data leak et assure cohérence train/test.

#### 3️⃣ **Sélection de Variables (SelectKBest)**
- Technique : Score F-regression
- Variables retenues : 50 (sur ~110 après encodage)
- Réduit bruit, accélère entraînement

#### 4️⃣ **Grid Search avec Cross-Validation**
- **Grille testée :**
  - Linear Regression
  - Ridge : α ∈ [0.1, 1, 10, 50]
  - Lasso : α ∈ [0.001, 0.01, 0.1, 1]
  - ElasticNet : α × l1_ratio
  - RandomForest : n_estimators, max_depth

- **Validation :** 5-fold CV avec scoring R²
- **Total :** 100+ configurations testées

#### 5️⃣ **Évaluation & Comparaison**
- Métriques : R², RMSE, MAE
- CV scores + écart-types (stabilité)
- Graphiques : R² | RMSE | CV Stability

#### 6️⃣ **Analysis**
- Prédictions vs Réel (scatter + diagonale)
- Résidus analysis (homoscédasticité)
- Feature Importance (top 20 variables)

---

## 📁 Structure du Projet

```
house-price-prediction/
│
├── house_prediction.ipynb          ← Notebook complet (EDA + ML)
├── train.csv                       ← Dataset d'entraînement
├── requirements.txt                ← Dépendances Python
│
├── models/
│   ├── best_model.pkl              ← Modèle sauvegardé (Random Forest)
│   └── model_comparison.csv        ← Résultats métriques
│
└── images/
    ├── model_comparison.png        ← Bar charts (R², RMSE, CV)
    ├── predictions_vs_real.png     ← Scatter plot
    └── residuals_analysis.png      ← Résidus analysis
```

---

## 🛠️ Technologies Utilisées

```python
📚 Data Processing
├── pandas          # Manipulation de données
├── numpy           # Calculs numériques

🤖 Machine Learning
├── scikit-learn    # Modèles, preprocessing, evaluation
│  ├── Pipeline
│  ├── ColumnTransformer
│  ├── GridSearchCV
│  ├── SelectKBest
│  └── [5 modèles : Linear, Ridge, Lasso, ElasticNet, RandomForest]

📊 Visualization
├── matplotlib      # Graphiques statiques
└── seaborn        # Graphiques avancés

💾 Model Persistence
└── joblib         # Sauvegarde/Chargement modèles
```

---

## 🚀 Guide d'Utilisation

### Installation

```bash
# Clone ou télécharge le projet
cd house-price-prediction

# Installe les dépendances
pip install -r requirements.txt

# Lance le notebook
jupyter notebook house_prediction.ipynb
```

### Utiliser le Modèle Entraîné

```python
import joblib
import pandas as pd

# Charge le modèle
best_model = joblib.load('models/best_model.pkl')

```

### Réentraîner le Modèle

Exécute simplement tout le notebook. La GridSearchCV s'exécutera et trouvera le meilleur modèle.

**Note :** GridSearchCV peut prendre 5-15 min selon le CPU.

---





### 1. Pourquoi Random Forest gagne



**Random Forest capture ces patterns** via décisions hiérarchiques.




## 📌 Limitations & Améliorations Futures

### Limitations Actuelles
-  Données 2010-2011 (prédictions actuelles ? à vérifier)
-  US only (Ames, Iowa) → Nécessite retraining pour autre région
-  Feature Engineering limité (pas de d nouvelles features créées)
-  Outliers non traités (maisons très atypiques mal prédites)

### Améliorations Possibles
1. **Feature Engineering avancé**
   - Interactions (Qualité × Localisation)
   - Polynomiales
   - Domain-specific (ex : âge de la maison)

2. **Modèles Avancés**
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks pour patterns très complexes
   - Stacking (ensemble de modèles)

3. **Gestion des Outliers**
   - IQR filtering
   - Robust scalers

4. **Données plus Riches**
   - Intégrer données externes (tendance marché, taux intérêt)
   - Plus de observations (1,460 → 10,000+)

---

5. **Réponse aux questions business**
 - Oui il existe des propriétés surévaluées/sous_évaluées
- Les 5 facteurs qui influencent le prix sont OverallQual,GrLivArea    GarageCars,YearBuilt,GarageArea 
- La localisation a un d'impact énorme sur le prix R² = 0,546 
-  le type de propriété qui se vends le plus chère en moyenne est le 2.5Fin 
- oui on peut faire confiance aux prédictions du modèle car il dépasse bien plus les kpi à atteindre (R² =0.89)

