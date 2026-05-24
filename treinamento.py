# -*- coding: utf-8 -*-
"""
Baseline TF-IDF + LinearSVC no BrScamsFacebook.

Protocolo identico ao BERTimbau:
  - Validacao cruzada estratificada k=5, seed=42
  - Relatorio por fold + relatorio consolidado (450 avaliacoes)
  - Comparativo final com classificador por maioria e BERTimbau
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report
)
from sklearn.dummy import DummyClassifier

# ------------------------------------------------------------------
# 1. Carregamento
# ------------------------------------------------------------------
df = pd.read_excel("BrScamsFacebook.xlsx")
X = df["Mensagem"].astype(str).values
y = df["Categoria"].values

categorias = sorted(df["Categoria"].unique())
print(f"Dataset: {len(df)} instancias | {len(categorias)} categorias")
print(f"Balanceamento: {dict(df['Categoria'].value_counts())}\n")

# ------------------------------------------------------------------
# 2. Pipeline
#    sublinear_tf=True: log-normalizacao padrao para SVM em texto
#    ngram_range=(1,2):  unigramas + bigramas
# ------------------------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("svm", LinearSVC(C=1.0, max_iter=2000, random_state=42))
])

# ------------------------------------------------------------------
# 3. k-fold estratificado (k=5) — identico ao BERTimbau
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_f1   = []
all_true  = []
all_pred  = []

print("=" * 60)
print("VALIDACAO CRUZADA ESTRATIFICADA (k=5)")
print("=" * 60)

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    pipeline.fit(X[train_idx], y[train_idx])
    y_pred = pipeline.predict(X[test_idx])

    f1 = f1_score(y[test_idx], y_pred, average="macro")
    fold_f1.append(f1)
    all_true.extend(y[test_idx])
    all_pred.extend(y_pred)

    print(f"  Fold {fold_idx}: F1-macro = {f1:.3f}")

mean_f1 = np.mean(fold_f1)
std_f1  = np.std(fold_f1)
print(f"\n  Media +/- DP : {mean_f1:.3f} +/- {std_f1:.3f}  <- resultado principal")

# ------------------------------------------------------------------
# 4. Relatorio consolidado (todas as 450 predicoes)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("RELATORIO CONSOLIDADO (450 avaliacoes, 75 por categoria)")
print("=" * 60)
print(classification_report(all_true, all_pred, target_names=categorias, digits=2))
print(f"Acuracia geral: {accuracy_score(all_true, all_pred):.1%}")

# ------------------------------------------------------------------
# 5. Classificador por maioria (lower bound de referencia)
# ------------------------------------------------------------------
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_f1 = []
for train_idx, test_idx in cv.split(X, y):
    dummy.fit(X[train_idx], y[train_idx])
    dummy_f1.append(f1_score(y[test_idx], dummy.predict(X[test_idx]), average="macro"))

# ------------------------------------------------------------------
# 6. Comparativo final
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARATIVO FINAL")
print("=" * 60)
print(f"  Classificador por maioria  : F1-macro ~{np.mean(dummy_f1):.3f}")
print(f"  TF-IDF + LinearSVC         : F1-macro  {mean_f1:.3f} +/- {std_f1:.3f}")
print(f"  BERTimbau fine-tuned       : F1-macro  0.763 +/- 0.034")
print(f"\n  Ganho BERTimbau vs baseline: +{0.763 - mean_f1:.3f} pp de F1-macro")
