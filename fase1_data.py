import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def load_all_files(folder="data/raw"):
    parquet_files = glob.glob(f"{folder}/*.parquet")
    csv_files = glob.glob(f"{folder}/*.csv")

    all_files = parquet_files + csv_files

    if not all_files:
        raise FileNotFoundError(
            f"Nenhum ficheiro encontrado em {folder}/\n"
            f"Certifica-te que os ficheiros estão lá dentro."
        )

    dfs = []
    for f in all_files:
        print(f"A ler: {f}")
        if f.endswith(".parquet"):
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(combined):,} linhas, {len(combined.columns)} colunas")
    return combined

df = load_all_files("data/raw")



ATTACK_KEYWORDS = ["DDoS", "Brute Force", "Bruteforce", "DoS", "Bot"]
RANDOM_STATE = 42
N_BENIGN = 50000
N_ATTACK = 50000

# Identificar a coluna Label
label_col = "Label"  # no CSE-CIC-IDS2018 é sempre esta
df[label_col] = df[label_col].astype(str).str.strip()

# Separar classes
benign_df = df[df[label_col].str.lower() == "benign"]
attack_df = df[df[label_col].str.contains(
    "|".join(ATTACK_KEYWORDS), case=False, na=False
)]

# Amostrar
benign_sample = benign_df.sample(n=N_BENIGN, random_state=RANDOM_STATE)
attack_sample = attack_df.sample(n=N_ATTACK, random_state=RANDOM_STATE)

# Juntar e baralhar
sampled = pd.concat([benign_sample, attack_sample], ignore_index=True)
sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)





# Criar target (0=benigno, 1=ataque)
y = (sampled[label_col].str.lower() != "benign").astype(int)
X = sampled.drop(columns=[label_col])

# Manter apenas colunas numéricas
X = X.select_dtypes(include=[np.number])

# Substituir inf/-inf por NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remover linhas com NaN
valid_mask = X.notna().all(axis=1)
X = X.loc[valid_mask].copy()
y = y.loc[valid_mask].copy()

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)






rf = RandomForestClassifier(
    n_estimators=150,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X, y)

# Ordenar por importância e pegar as top 15
importance = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

top15 = importance.head(15).index.tolist()
X_top = X[top15].copy()

# Guardar lista de features e importâncias
importance.to_csv("data/processed/feature_importance.csv", header=["importance"])
joblib.dump(top15, "models/top15_features.pkl")






scaler = MinMaxScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_top),
    columns=top15
)

# Guardar scaler para usar nas fases seguintes
joblib.dump(scaler, "models/minmax_scaler.pkl")

# Guardar CSV final com target
final_df = X_scaled.copy()
final_df["target"] = y.values
final_df.to_csv("data/processed/cicids2018_clean_top15.csv", index=False)







X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred,
      target_names=["Benign", "Attack"]))




# Guardar modelo
joblib.dump(model, "models/xgb_baseline.pkl")

# Guardar datasets de treino/teste
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)