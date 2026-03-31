# 📚 Guide Complet de Compréhension du Code ML

> Ce guide t'aide à **comprendre en profondeur** et **mémoriser** les concepts clés du notebook

---

## 📖 Table des matières
1. [Flux global du projet](#flux-global)
2. [Pipeline & ColumnTransformer](#pipeline--columntransformer)
3. [GridSearchCV - Comment ça marche](#gridsearchcv)
4. [Les Métriques - Interprétation](#metriques)
5. [Visualisations & Graphiques](#visualisations)
6. [Preprocessing détaillé](#preprocessing)

---

## 🔄 Flux Global du Projet {#flux-global}

```
DONNÉES BRUTES (train.csv)
        ↓
📊 CHARGEMENT & EXPLORATION
        ↓
🧹 TRAIN/TEST SPLIT (80/20)
        ↓
⚙️ CREATION DU PREPROCESSOR (ColumnTransformer)
        ↓
🤖 CREATION DU PIPELINE (Preprocessor + SelectKBest + Model)
        ↓
🔍 GRIDSEARCHCV (teste toutes les combinaisons)
        ↓
🏆 MEILLEUR MODELE IDENTIFIE
        ↓
📊 COMPARAISON DE TOUS LES MODELES
        ↓
📈 VISUALISATIONS & METRIQUES
        ↓
💾 SAUVEGARDE DU MODELE & RESULTATS
```

---

## 🔧 Pipeline & ColumnTransformer {#pipeline--columntransformer}

### 🎯 Qu'est-ce qu'un Pipeline ?

Un **Pipeline** est une **chaîne de transformation linéaire** où chaque étape :
1. Reçoit les données de l'étape précédente
2. Les transforme
3. Les passe à l'étape suivante

**Avantages :**
- ✅ Evite les **data leaks** (pas de leakage entre train/test)
- ✅ Réutilisable (on peut sauvegarder tout le pipeline en une seule variable)
- ✅ Clair et lisible

### 📐 Structure principale

```python
pipeline = Pipeline([
    ('preprocessor', preprocessor),              # Étape 1 : Nettoyage + encodage
    ('feature_selection', SelectKBest(...)),     # Étape 2 : Sélection des meilleures variables
    ('model', LinearRegression())                # Étape 3 : Entraînement du modèle
])
```

**Le flux des données :**
```
Données brutes (X_train)
    ↓
[Preprocessor] → Données nettoyées, centrées, encodées
    ↓
[SelectKBest] → Les 50 meilleures variables seulement
    ↓
[Model] → Entraîné & prêt à prédire
```

### 🏗️ ColumnTransformer - Transformer implicitement

Le **ColumnTransformer** est un "preprocessor intelligent" qui **appelle différentes transformations selon le type de colonne** :

```python
preprocessor = ColumnTransformer(
    transformers=[
        # Pipeline pour colonnes NUMERIQUES
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),    # Remplace NaN par la médiane
            ('scaler', StandardScaler())                      # Centre & normalise (μ=0, σ=1)
        ]), num_features),
        
        # Pipeline pour colonnes CATEGORIQUES
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Remplace NaN par le mode
            ('encoder', OneHotEncoder(drop='first'))               # Convertit texte → nombres
        ]), cat_features)
    ]
)
```

**Exemple concret :**

Avant preprocessing :
```
Age  | Maison_Type | Prix
-----|-------------|------
25   | Studio      | 100k
NaN  | Villa       | 200k
35   | Studio      | 150k
```

Après preprocessing :
```
Age_scaled | Maison_Type_Studio | Maison_Type_Villa | Prix
-----------|-------------------|-------------------|----- (y)
-0.5       | 1                  | 0                 | 100k
0.0        | 0                  | 1                 | 200k
0.8        | 1                  | 0                 | 150k
```

**Ce qui s'est passé :**
- ✅ Age=NaN → remplacé par la médiane (25)
- ✅ Age → centre & échelonné (StandardScaler)
- ✅ Maison_Type="Studio" → colonne numérique 1 ou 0 (OneHotEncoder)

---

## 🔍 GridSearchCV - Comment ça marche {#gridsearchcv}

### 💡 L'idée clé

**On veut trouver les MEILLEURS hyper-paramètres pour notre pipeline.**

Des questions :
- Ridge : alpha = 0.1 ou 1.0 ou 10.0 ?
- RandomForest : n_estimators = 100 ou 200 ?
- SelectKBest : k = 30 ou 50 variables ?

**GridSearchCV teste TOUTES les combinaisons** et dit laquelle est la meilleure.

### 🎰 Exemple simplifié

Imagine cette grille très petite :

```python
param_grid = [
    {
        'model': [Ridge()],
        'model__alpha': [1.0, 10.0]
        'feature_selection__k': [30, 50]
    }
]
```

GridSearchCV va tester **2 × 2 = 4 combinaisons** :

| Num | Alpha | k | Score CV |
|-----|-------|---|----------|
| 1   | 1.0   | 30| 0.82     |
| 2   | 1.0   | 50| **0.84** |
| 3   | 10.0  | 30| 0.79     |
| 4   | 10.0  | 50| 0.81     |

→ **La combinaison (alpha=1.0, k=50) gagne** !

### 🔧 Comment ça marche en coulisse

```python
grid = GridSearchCV(
    pipeline,           # Le pipeline à optimiser
    param_grid,         # Grille des paramètres
    cv=5,               # 5-fold cross-validation
    scoring='r2',       # Métrique : R²
    n_jobs=-1,          # Utilise tous les CPU
    verbose=1           # Affiche la progression
)

grid.fit(X_train, y_train)  # Lance la recherche
```

**Process (pour chaque combinaison) :**

```
1. Divise X_train en 5 sous-ensembles (fold 1, 2, 3, 4, 5)
2. Pour chaque fold :
   - Entraîne le pipeline sur 4 folds
   - Évalue sur le 5ème fold
3. Calcule le score moyen sur les 5 évaluations
4. Sauvegarde ce score comme le score CV pour cette combinaison
5. Passe à la combinaison suivante
```

**Résultat :**
- `grid.best_estimator_` : le pipeline réentraîné avec les MEILLEURS paramètres
- `grid.best_params_` : les paramètres optimaux
- `grid.best_score_` : le score CV moyen avec ces paramètres

---

## 📊 Les Métriques - Interprétation {#metriques}

### 1️⃣ **RMSE (Root Mean Squared Error)**

```
RMSE = √(Σ(y_réel - y_prédit)² / n)
```

**Intuition :** "En moyenne, mon modèle se trompe de combien ?"

**Exemple :**
- Si RMSE = $30,000 → Le modèle se trompe en moyenne de 30k$ sur un prix
- Plus bas = mieux ✅

**Attention :** pénalise beaucoup les GRANDS écarts (erreurs extrêmes pèsent lourd).

---

### 2️⃣ **MAE (Mean Absolute Error)**

```
MAE = Σ|y_réel - y_prédit| / n
```

**Intuition :** "Erreur moyenne en valeur absolue"

**Exemple :**
- MAE = $20,000 → En moyenne, j'étais à $20k du prix réel
- Plus bas = mieux ✅

**Vs RMSE :** MAE penalise moins les écarts extrêmes (plus "robuste").

**Comparaison :**
| Prédictions | RMSE    | MAE  | Observation |
|-------------|---------|------|-------------|
| [±10k, ±10k, ±80k] | $30k    | $33k | RMSE penalise l'erreur extrême (+80k) |
| [±10k, ±10k, ±80k] | $33k    | $33k | MAE less affected |

---

### 3️⃣ **R² (Coefficient de Détermination)**

```
R² = 1 - (Σ(y_réel - y_prédit)² / Σ(y_réel - y_moyen)²)
```

**Intuition :** "Quelle fraction de la variance originale mon modèle explique-t-il ?"

**Interprétation :**
- R² = 0.95 → Mon modèle explique **95% de la variabilité** des prix
- R² = 0.50 → Mon modèle explique **50% de la variabilité** (moyen)
- R² = 0.00 → Mon modèle est aussi mauvais que la **moyenne globale**
- R² < 0 → Mon modèle pire que juste donner la moyenne ❌

**Dans ton projet :**
- Objectif : R² > 0.85 (Tu dois expliquer 85% de la variabilité)

---

### 4️⃣ **CV Scores (Cross-Validation)**

```
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
# cv_scores = [0.84, 0.86, 0.83, 0.85, 0.82]

CV R² Mean = 0.84
CV R² Std  = 0.016
```

**Intuition :**
- **Mean** : Score moyen du modèle sur les 5 folds
- **Std** : Variabilité entre les folds (stabilité du modèle)
  - Std bas (ex: 0.01) = modèle stable, performant de façon consistent
  - Std élevé (ex: 0.10) = modèle instable, performance variable selon les données

**Pourquoi ça ? Éviter l'overfitting !**
- Un modèle peut avoir un excellent score sur TEST mais mauvais sur données nouvelles.
- La CV montre comment le modèle se généralise sur DIFFERENTES portions des données.

---

## 📈 Visualisations & Graphiques {#visualisations}

### Figure 1 : Comparaison des Modèles (3 sous-graphiques)

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# → Crée une figure avec 1 ligne et 3 colonnes de graphiques
```

**Sous-graphique 1 : R²**
```
Random Forest  ████████░  0.88
Ridge          ██████░    0.75
Lasso          █████░     0.70
             R² Score
             ↑ Trait rouge = objectif (0.85)
```
- Quel modèle explique le plus de variance ?
- Lequel dépasse l'objectif ?

**Sous-graphique 2 : RMSE**
```
Linear Regr    ██████████░    32000$
Lasso          ████████░      25000$
RandomForest   ████░          18000$
             RMSE ($)
             ↑ Trait rouge = objectif (<30k)
```
- Quel modèle a les erreurs les plus petites ?

**Sous-graphique 3 : CV Scores**
```
RandomForest   ██████████  0.85 ± 0.02
Ridge          ██████░     0.72 ± 0.05
Lasso          █████░      0.69 ± 0.08
             R² (CV) with error bars
```
- Les barres d'erreur montrent la **stabilité** (petite barre = stable, grande = instable)

---

### Figure 2 : Prédictions vs Réel

```
      Prix Prédit
      ^
 300k |          .
      |      . .
 250k |     . . .
      |   . . . .
 200k |  . . . .        ← Trait rouge = parfait
      | . . . .           (y_réel = y_prédit)
 150k |. . . .
      |. . .
 100k |. .
      |.
   0k +--+--+--+--+--+
      0k 100k 200k 300k
      ↑ Prix Réel
```

**Ce qu'on cherche :**
- Points proches de la **diagonale rouge** = bonnes prédictions
- Points loin = **erreurs importantes**
- Pas de **tendan extrême** (ex : systématiquement prédire trop haut)

---

### Figure 3 : Analyse des Résidus

**Sous-figure 1 : Résidus vs Prédictions**
```
Résidus ($)
      +10k |     . .
           |   .   .
         0 |.. . . ...  ← La ligne d'équilibre (résidu = 0)
           | .  .  .
     -10k |    .   .
           +-----------
           Prédictions ($)
```
**Idéal :** nuage centré autour de 0 (pas de pattern)
**Mauvais :** nuage qui monte/descend (modèle biaisé)

**Sous-figure 2 : Histogramme des Résidus**
```
Fréquence
    |     █
    |    ██
    |   ███      ← Forme en cloche idéale
    |  █████
    | ██████
    |█████████
    +----------
    -10k 0k +10k
    ↑ Résidus ($)
```
**Idéal :** courbe en cloche centrée à 0 (distribution normale)
**Problème :** si très asymétrique = le modèle a un biais systématique

---

## 🧹 Preprocessing Détaillé {#preprocessing}

### Pourquoi le preprocessing est CRITIQUE

**Sans :**
```
Age (0-100) | Revenu (0-500k) | Surface (20-500 m²)
25          | 50000           | 80
30          | 75000           | 120
21          | 35000           | 45
```

**Problem :** Revenu est sur une échelle **BEAUCOUP PLUS GRANDE** que Age !
→ Le modèle pensera que Revenu est plus important juste par sa magnitude.
→ Résultat biaisé ❌

---

### SimpleImputer - Geérer les valeurs manquantes

```python
SimpleImputer(strategy='median')  # Numérique
SimpleImputer(strategy='most_frequent')  # Catégorique
```

**Avant :**
```
Age
25
NaN
35
NaN
40
```

**Après :**
```
Age (médiane = 33.3)
25
33
35
33
40
```

- `median` : moins sensible aux valeurs extrêmes que la moyenne
- `most_frequent` : pour les catégories, prend le mode le plus courant

---

### StandardScaler - Normaliser les nombres

```python
StandardScaler()  # Transform to μ=0, σ=1
```

**Avant :**
```
Age : [25, 30, 35, 40, 45]  (μ=35, σ≈7.9)
```

**Après :**
```
Age_scaled : [-1.27, -0.63, 0, 0.63, 1.27]  (μ=0, σ=1)
```

**Formula :** `x_scaled = (x - mean) / std`

**Pourquoi ?**
- Met tout sur la même échelle (centré, réduit)
- Des algos comme Ridge/Lasso dépendent de l'échelle → StandardScaler les aide
- RandomForest ne l'exige pas, mais ça aide tous les modèles à converger plus vite

---

### OneHotEncoder - Transformer le texte en nombres

```python
OneHotEncoder(drop='first')
```

**Avant :**
```
Maison_Type
Studio
Villa
Studio
Appartement
```

**Après (avec drop='first') :**
```
Villa | Appartement
0     | 0           ← Studio (dropped)
1     | 0           ← Villa
0     | 0           ← Studio
0     | 1           ← Appartement
```

**Pourquoi `drop='first'` ?**
- Évite la colinéarité parfaite (sinon : Villa + Appartement + Studio = 1 toujours)
- Réduit le nombre de colonnes inutiles

---

## 🎯 Synthèse - Ce à Retenir

| Concept | Rôle | Output |
|---------|------|--------|
| **ColumnTransformer** | Nettoie données num/cat | Données transformées |
| **SelectKBest** | Sélectionne les 50 meilleures variables | Variables pertinentes |
| **Pipeline** | Enchaîne les étapes | Objet réutilisable complet |
| **GridSearchCV** | Teste toutes les combinaisons de paramètres | Meilleurs hyper-paramètres |
| **Métriques** | Évalue la performance | Scores (R², RMSE, MAE) |
| **Visualisations** | Montre visuellement quoi améliorer | Graphiques d'analyse |

---

## 🚀 Comment l'apprendre par la pratique

1. **Modifiez `param_grid`** : Ajoute plus de valeurs d'alpha, autres modèles
2. **Change le preprocessing** : Teste `StandardScaler()` vs sans. Vois la différence
3. **Modifiez `k` du SelectKBest** : 20, 50, 'all' → compare les résultats
4. **Affiche des intermédiaires** : `print(X_train.shape)` après chaque étape du pipeline

---

**Bonne compréhension !** 🎓
