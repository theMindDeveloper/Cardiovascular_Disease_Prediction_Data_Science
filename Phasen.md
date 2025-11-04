# Forschungsfrage
Kann man auf Basis von Gesundheits- und Lifestyle-Daten mit Machine Learning vorhersagen, ob eine Person an einer Herz-Kreislauf-Erkrankung leidet?
## Thema
Einfluss von Lifestyle auf Krankheit (CVD)

## Hypothesen
-  Mit Machine-Learning-Modellen lassen sich Korrelationen und Zusammenhänge zwischen Lebensstil-/Gesundheitsdaten und dem Vorhandensein von Herz-Kreislauf-Erkrankungen erkennen.
 --Weitere Hypothesen “Bestimmte Faktoren (z.B. Rauchen, hoher Blutdruck) haben den größten Einfluss auf das Erkrankungsrisiko.”
-  
-  

## Daten
Datensatz: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset?resource=download

# Phasen

### Phase 0 — Setup (Team & Repo) (22.08 - 29.08)
1. Rollen klären  
    - Wer übernimmt welche Rolle? (Rollen können rotieren, z. B. erst alle zusammen EDA, dann Aufteilung)

2. Aufteilung in Hauptbereiche / Rollen
    - EDA / Cleaning   (ALLE ZUSAMMEN)
      - Aufgaben: Daten explorieren, fehlende Werte behandeln (z. B. Alter in Kategorien)  
      - Ziele: Prüfen, ob Datensatz unterschiedliche Klassen hat (z. B. mehr Gesunde als Patienten)
    - Modeling  
      - Aufgaben: Algorithmen auswählen für Vorhersage von CVD  
         - Vorschläge: Logistic Regression (Baseline), Random Forest (Feature-Importance)  
      - Ziele: Daten vorbereiten und erste Modelle trainieren (verschiedene Modelle, Feature-Kombinationen)  
         - Zu untersuchen: Alter, Gewicht, Rauchen, Alkohol etc.  
      - Tools: `pandas`, `numpy`, ...
    - Evaluation / Interpretation  
      - Aufgaben: Metriken wählen; Modelle vergleichen (verschiedene Feature-Kombinationen)  
      - Ziele: Ergebnisse interpretieren
    - Writing / Slides  
      - Aufgaben: Dokumentation (README, Notebooks), Präsentation vorbereiten (Zielgruppe z. B. Ärzt:innen, Data-Science-Kollegen, Mitstudierende)  
      - Ziele: Forschungsfrage und Ergebnisse klar kommunizieren; Slides z. B. „Wie unser Modell Lifestyle-Faktoren mit CVD verknüpft“

3. Repo einrichten  
4. Daten beschaffen (Datensatz bereits vorhanden)  
5. Erste Schritte im Notebook  
    - Beispiel: Zwei Variablen wählen und auswerten (Alter und Smoking → erhöhte Rate?)

### Phase 1 — Problem & Daten (Initial)
- **Ziel:** Forschungsfrage formulieren, Daten herunterladen und initial laden.
- **Aufgaben**
  - Forschungsfrage & Hypothesen klar formulieren.
  - Dataset von Kaggle speichern: `data/raw/`.
  - Erste Übersicht: `df.info()`, `df.describe()`, `df.head()`.
- **Tools / Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Output:** `notebooks/01_overview.ipynb` mit Kurzdokumentation.

---

### Phase 2 — Explorative Datenanalyse (EDA)
- **Ziel:** Daten verstehen, Probleme & erste Muster erkennen.
- **Aufgaben**
  - Variablen umrechnen: `age_days → age_years`.
  - Neue Features: `BMI = weight / (height/100)**2`, `pulse_pressure = ap_hi - ap_lo`.
  - Verteilungen prüfen, Zielklassen-Verhältnis, Missingness, Ausreißer.
  - Visuals: Histogramme, Boxplots, Heatmap (Korrelationen), Target-Vergleiche.
- **Tools / Libraries:** `pandas`, `seaborn`, `matplotlib`, optional `pandas_profiling`
- **Output:** `notebooks/02_EDA.ipynb` mit Grafiken & Beobachtungen.

---

### Phase 3 — Cleaning & Feature Engineering
- **Ziel:** Saubere, modellierbare Daten erzeugen.
- **Aufgaben**
  - Outlier-Regeln dokumentieren (z. B. `ap_hi ∈ [80,240]`, `ap_lo ∈ [40,140]`, `ap_hi > ap_lo`).
  - Umgang mit fehlenden Werten / „Unknown“ (z. B. `smoking_status`).
  - Encoding: One-Hot / Ordinal (z. B. `cholesterol`, `gluc`).
  - Feature Scaling vorbereiten (Scaler nur später auf Training fitten).
  - Finale Tabelle speichern: `data/processed/clean_data.csv`.
- **Tools / Libraries:** `pandas`, `scikit-learn` (`SimpleImputer`, `OneHotEncoder`, `ColumnTransformer`)
- **Output:** `notebooks/03_cleaning.ipynb`, `data/processed/clean_data.csv`.

---

### Phase 4 — Train/Test-Split (richtig positioniert)
- **Ziel:** Robuste Evaluation sicherstellen.
- **Aufgaben**
  - Stratified Split: `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`.
  - Speichere Sets oder DataFrame-Indizes für Reproduzierbarkeit.
- **Tools / Libraries:** `scikit-learn.model_selection`
- **Output:** `X_train, X_test, y_train, y_test` (persistiert optional in `data/processed/`).

---

### Phase 5 — Baseline-Modelle
- **Ziel:** Einfache, erklärbare Benchmarks.
- **Modelle**
  - `DummyClassifier` (Majority) → Sanity check.
  - `LogisticRegression` (baseline, interpretierbar).
- **Metriken:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- **Tools:** `scikit-learn` (`LogisticRegression`, `metrics`)
- **Output:** `notebooks/04_baselines.ipynb` mit Ergebnis-Tabelle und ersten Plots.

---

### Phase 6 — Fortgeschrittene Modelle & Tuning
- **Ziel:** Leistungsfähige, robuste Modelle entwickeln.
- **Modelle & Vorgehen**
  - Random Forest, XGBoost / LightGBM.
  - 5-fold Stratified CV; `GridSearchCV` oder `RandomizedSearchCV`.
  - Hyperparameter: `n_estimators`, `max_depth`, `learning_rate`, `subsample`.
- **Tools:** `sklearn`, `xgboost` / `lightgbm`, `joblib` (für Persistenz)
- **Output:** `notebooks/05_modeling.ipynb` mit CV-Ergebnissen und finalem Modell.

---

### Phase 7 — Interpretation & Explainability
- **Ziel:** Modelle verständlich machen — welche Features sind wichtig?
- **Aufgaben**
  - Feature Importance (Tree: `.feature_importances_`; LR: `.coef_`).
  - SHAP-Summary / SHAP-Dependence (empfohlen für GBDT).
  - ROC, PR Curves, Confusion Matrix visualisieren.
  - (Optional) Partial Dependence / ICE für Top-Features.
- **Tools:** `shap`, `sklearn`, `matplotlib`, `seaborn`
- **Output:** `notebooks/06_interpretation.ipynb`, `reports/figures/`.

---

### Phase 8 — Robustness, Fairness & Limitations
- **Ziel:** Qualität, Fairness und Grenzen transparent machen.
- **Aufgaben**
  - Check subgroup performance (z. B. nach `gender`): precision/recall per Gruppe.
  - Calibration plot (Reliability curve); ggf. `CalibratedClassifierCV`.
  - Sensitivity analysis (z. B. andere Outlier-Regeln).
  - Ethik: Datenquelle, Bias, keine medizinischen Aussagen.
- **Tools:** `sklearn.calibration`, `pandas`
- **Output:** Kurzkapitel in Paper & Slides.

---

### Phase 9 — Finalisierung & Deliverables
- **Deliverables**
  - 4–6 aussagekräftige Plots (EDA, ROC/PR, ConfMat, Feature Importance, Calibration).
  - Sauberes Notebook(s) mit narrativem Fluss.
  - `requirements.txt` / `environment.yml`.
  - Seminararbeit (6–8 Seiten) oder PDF in `reports/`.
  - Präsentation (Slides).


- [ ] Dataset heruntergeladen (`data/raw/`)
- [ ] EDA Notebook fertig (Plots + Beobachtungen)
- [ ] Cleaning rules dokumentiert und angewendet
- [ ] Train/Test split erzeugt (stratified)
- [ ] Baseline (Dummy + LR) trainiert und evaluiert
- [ ] 1–2 starke Modelle trainiert & validiert (RF/XGB)
- [ ] Feature-Importance + SHAP erstellt
- [ ] Robustness & subgroup checks durchgeführt
- [ ] Report + Notebooks + Slides erstellt