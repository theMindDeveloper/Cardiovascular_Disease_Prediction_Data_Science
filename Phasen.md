# Forschungsfrage
Kann man mit gesundheitlichen und Lifestyle-Daten mittels Machine Learning vorhersagen, ob eine Person an Cardiovascular Disease (Herz-Kreislauf-Erkrankung) leidet?

## Thema
Lifestyle — Auswirkung auf Erkrankung

## Hypothesen
-  
-  
-  

## Daten
Datensatz: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset?resource=download

# Phasen

## Phase 0 — Setup (Team & Repo) (22.08 - 29.08)
1. Rollen klären  
    - Wer übernimmt welche Rolle? (Rollen können rotieren, z. B. erst alle zusammen EDA, dann Aufteilung)

2. Aufteilung in Hauptbereiche / Rollen
    - EDA / Cleaning  
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

## Phase 1 — Problem & Data
- Ziel: Vorhersage von cardio aus Lifestyle / Vitaldaten  
- Train/holdout split: stratified 80/20 (oder 70/30) mit fixem Random Seed

## Phase 2 — EDA (Exploratory Data Analysis)
- Alter konvertieren (Tage → Jahre), BMI berechnen, Pulse Pressure (ap_hi - ap_lo) berechnen  
- Verteilungen prüfen, Missingness prüfen (meist minimal), auffällige Ausreißer identifizieren (z. B. diastolisch > systolisch, ap_hi > 250, ap_lo < 40)  
- Einfache Visuals: Histogramme, Boxplots, Paarweise Korrelationen, Target vs. Schlüsselvariablen

## Phase 3 — Cleaning & Feature-Preparation
- Outlier-Regeln dokumentieren, z. B. behalten: ap_hi ∈ [80, 240], ap_lo ∈ [40, 140], und ap_hi > ap_lo  
- Kategorische Variablen encoden (gender, cholesterol, gluc) → one-hot oder ordinal je nach Bedeutung  
- Scaling (nur falls linear/SVM): `StandardScaler` für numerische Spalten

## Phase 4 — Baselines
- DummyClassifier (majority) → Sanity check  
- Logistic Regression (mit/ohne Scaling) → interpretierbare Baseline  
- Metriken aufnehmen: Accuracy, F1, ROC-AUC, PR-AUC (PR-AUC bei leichter Klassenungleichheit hilfreich)

## Phase 5 — Stärkere Modelle
- Random Forest, Gradient Boosted Trees (z. B. XGBoost / LightGBM)  
- 5-fache stratified CV auf Trainingssatz; einfache Hyperparameter-Suche (max_depth, n_estimators, learning_rate)  
- Vergleich auf gehaltenem Testset (kein Peeking)

## Phase 6 — Interpretation
- Feature-Importance: Gain / SHAP für Tree-Modelle; Koeffizienten für LR  
- Partial Dependence oder einfache ICE für Top-2/3 Features (optional)  
- Kalibrationsplot (Reliability Curve) zur Darstellung der Wahrscheinlichkeitsqualität (optional)

## Phase 7 — Robustness & Ethics
- Klassenbalance nach Cleaning prüfen; ggf. `class_weight='balanced'` verwenden und dokumentieren  
- Kurzer Hinweis zu Bias (z. B. Performance nach Geschlecht)  
- Limitierungen: self-reported Lifestyle, Single-Snapshot, keine medizinische Beratung

## Phase 8 — Deliverables
- Visuals (4–6): EDA (2), ROC/PR-Curves, Confusion Matrix, Feature Importance, Kalibration (optional)  
- Paper (6–8 Seiten): Problem, Daten, Methoden, Ergebnisse, Interpretation, Limitationen  
- Sauberes Notebook mit narrativem Fluss + `requirements.txt`
