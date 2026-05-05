import os, time, joblib
import numpy as np
import pandas as pd
import shap, lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import norm


# ──────────────────────────────────────────────
#  1. CARREGAMENTO DE DADOS
# ──────────────────────────────────────────────


os.makedirs("results", exist_ok = True)

model   = joblib.load("models/xgb_baseline.pkl")
top15   = joblib.load("models/top15_features.pkl")
X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")
x_real  = pd.read_csv("results/x_real_attack.csv")
x_adv   = pd.read_csv("results/x_adv_attack.csv")

# Latências da Fase 2
lat = pd.read_csv("results/latency_metrics.csv")
shap_mean_ms = lat.loc[lat["method"] == "SHAP", "mean_ms"].values[0]
lime_mean_ms = lat.loc[lat["method"] == "LIME", "mean_ms"].values[0]


# ──────────────────────────────────────────────
#  2. Jaccard — SHAP e LIME
# ──────────────────────────────────────────────


shap_explainer = shap.TreeExplainer(model)
shap_real_vals = shap_explainer(x_real).values[0]
shap_adv_vals  = shap_explainer(x_adv).values[0]

def jaccard(set_a, set_b):
    return len(set_a & set_b) / len(set_a | set_b)

def top_k_set(vals, features, k=5):
    return set(pd.Series(np.abs(vals), index=features)
               .sort_values(ascending=False).head(k).index)

# Jaccard SHAP
jaccard_shap = jaccard(top_k_set(shap_real_vals, top15),
                       top_k_set(shap_adv_vals,  top15))

# Jaccard LIME
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data = X_train.values,
    feature_names = top15,
    class_names   = ["Benign", "Attack"],
    mode          = "classification",
    random_state  = 42
)
lime_real_exp = lime_explainer.explain_instance(x_real.iloc[0].values, model.predict_proba, num_features=15)
lime_adv_exp  = lime_explainer.explain_instance(x_adv.iloc[0].values,  model.predict_proba, num_features=15)

def lime_top_set(exp, k=5):
    s = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)
    return set([f[0].split(" ")[0] for f in s[:k]])

jaccard_lime = jaccard(lime_top_set(lime_real_exp), lime_top_set(lime_adv_exp))

print(f"Jaccard SHAP: {jaccard_shap:.4f}")
print(f"Jaccard LIME: {jaccard_lime:.4f}")


# ──────────────────────────────────────────────
#  3. Distância Euclidiana 
# ──────────────────────────────────────────────


euclidean_shap = norm(shap_real_vals - shap_adv_vals)

lime_r = np.array([v for _, v in lime_real_exp.as_list()])
lime_a = np.array([v for _, v in lime_adv_exp.as_list()])
euclidean_lime = norm(lime_r[:min(len(lime_r),len(lime_a))] - lime_a[:min(len(lime_r),len(lime_a))])

print(f"Euclidiana SHAP: {euclidean_shap:.4f}")
print(f"Euclidiana LIME: {euclidean_lime:.4f}")


# ──────────────────────────────────────────────
#  4. Throughput 
# ──────────────────────────────────────────────


N = 100

t0 = time.perf_counter()
for i in range(N):
    _ = shap_explainer(X_test.iloc[[i % len(X_test)]])
shap_throughput = N / (time.perf_counter() - t0)

t0 = time.perf_counter()
for i in range(N):
    _ = lime_explainer.explain_instance(
        X_test.iloc[i % len(X_test)].values,
        model.predict_proba, num_features=15
    )
lime_throughput = N / (time.perf_counter() - t0)

print(f"SHAP: {shap_throughput:.1f} pkt/s")
print(f"LIME: {lime_throughput:.1f} pkt/s")


# ──────────────────────────────────────────────
#  5. Tabela comparativa final + gráficos
# ──────────────────────────────────────────────


# ── Tabela CSV ──
final_table = pd.DataFrame({
    "Métrica"  : ["Latência (ms/pkt)", "Throughput (pkt/s)",
                  "Jaccard Top-5", "Distância Euclidiana"],
    "SHAP"     : [shap_mean_ms, shap_throughput, jaccard_shap, euclidean_shap],
    "LIME"     : [lime_mean_ms, lime_throughput, jaccard_lime, euclidean_lime],
})
final_table.to_csv("results/benchmark_final_table.csv", index=False)

# ── Radar Chart ──
max_tput   = max(shap_throughput, lime_throughput)
max_euclid = max(euclidean_shap, euclidean_lime)

shap_r = [shap_throughput/max_tput, jaccard_shap, 1-(euclidean_shap/max_euclid)]
lime_r = [lime_throughput/max_tput, jaccard_lime, 1-(euclidean_lime/max_euclid)]
angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist() + [0]
shap_r += [shap_r[0]]; lime_r += [lime_r[0]]

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True), facecolor="#0d1117")
ax.set_facecolor("#161b22")
ax.plot(angles, shap_r, color="#58a6ff", lw=2, label="SHAP")
ax.fill(angles, shap_r, color="#58a6ff", alpha=0.2)
ax.plot(angles, lime_r, color="#3fb950", lw=2, label="LIME")
ax.fill(angles, lime_r, color="#3fb950", alpha=0.2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Velocidade","Jaccard","Resist. Euclid."], color="#e6edf3")
ax.set_yticklabels([]); ax.grid(color="#30363d")
ax.legend(loc="upper right", bbox_to_anchor=(1.2,1.1),
          facecolor="#1c2128", edgecolor="#30363d", labelcolor="#e6edf3")
ax.set_title("Resiliência XAI — SHAP vs LIME", color="#e6edf3", fontweight="bold", pad=20)
plt.savefig("results/radar_benchmark.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()