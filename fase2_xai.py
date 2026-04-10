import os, time, joblib
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────
#  1. CARREGAMENTO DOS OUTPUTS
#     Carrega o modelo, features e datasets processados.
# ──────────────────────────────────────────────────────

# Confirma se a pasta 'results' existe. Caso contrário, cria-a.
os.makedirs("results", exist_ok = True)

# Carrega o modelo.
model = joblib.load("models/xgb_baseline.pkl")

# Carrega as features.
top15 = joblib.load("models/top15_features.pkl")

# Carrega os datasets.
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test_df = pd.read_csv("data/processed/y_test.csv")
y_test = y_test_df.iloc[:, 0]

# ───────────────────────────────────────────────────────────────
#  2. CONFIGURAÇÃO DO TREESHAP
#     Implementa o TreeSHAP, que ajuda na otimização dos modelos.
# ───────────────────────────────────────────────────────────────

# Configura o SHAP para o modelo XGBoost.
shap_explainer = shap.TreeExplainer(model)

# Teste rápido ao SHAP, para garantir que está a funcionar.
sample = X_test.iloc[[0]]
shap_values = shap_explainer(sample)
print(f"SHAP OK - Shape: {shap_values.values.shape}")

# ───────────────────────────────────────────────────────────
#  3. CONFIGURAÇÃO DO LIME
#     Configura-se o LIME, para explicar previsões do modelo.
# ───────────────────────────────────────────────────────────

# Configura o LIME para obter os dados de treino.
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data = X_train.values,
    feature_names = top15,
    class_names = ["Benign", "Attack"],
    mode = "classification",
    random_state = 42
)

# Teste rápido ao LIME, para garantir que está a funcionar.
lime_exp = lime_explainer.explain_instance(
    data_row = X_test.iloc[0].values,
    predict_fn = model.predict_proba,
    num_features = 15
)
print(f"LIME OK - {len(lime_exp.as_list())} features explained")

# ──────────────────────────────────────────────────────────────────
#  4. LATÊNCIA DO SHAP
#     Cronometra 100 explicações e calcula a média em milissegundos.
# ──────────────────────────────────────────────────────────────────

N = 100 # Número de explicações para medir a latência.
shap_times = []

# Seleciona uma linha de teste para explicar, e repete o processo N vezes para obter uma média.
for i in range(N):
    instance = X_test.iloc[[i % len(X_test)]]
    start = time.perf_counter()
    _ = shap_explainer(instance)
    shap_times.append((time.perf_counter() - start) * 1000)

# Calcula a média e o desvio padrão dos tempos do SHAP.
shap_mean_ms = np.mean(shap_times)
shap_std_ms = np.std(shap_times)

# Ponto Zero é o tempo base que o SHAP demora a explicar 1 pacote sem qualquer perturbação.
print(f"Ponto Zero SHAP = {shap_mean_ms:.2f} ms (± {shap_std_ms:.2f} ms)")

# ──────────────────────────────────────────────────
#  5. LATÊNCIA DO LIME
#     Mesmo processo de medição, só que para o LIME.
# ──────────────────────────────────────────────────

lime_times = []

# Seleciona uma linha de teste para explicar, e repete o processo N vezes para obter uma média.
for i in range(N):
    instance = X_test.iloc[i % len(X_test)].values
    start = time.perf_counter()
    _ = lime_explainer.explain_instance(
        data_row = instance,
        predict_fn = model.predict_proba,
        num_features = 15
    )
    lime_times.append((time.perf_counter() - start) * 1000)

# Calcula a média e o desvio padrão dos tempos do LIME.
lime_mean_ms = np.mean(lime_times)
lime_std_ms = np.std(lime_times)

print(f"Ponto Zero LIME = {lime_mean_ms:.2f} ms ± {lime_std_ms:.2f} ms")

# ─────────────────────────────────────────────────
#  6. ARMAZENAMENTO DAS MÉTRICAS
#     Guarda num ficheiro CSV os tempos registados.
# ─────────────────────────────────────────────────

# Armazena as métricas de latência num DataFrame e depois num CSV.
latency_df = pd.DataFrame({
    "method" : ["SHAP", "LIME"],
    "mean_ms" : [shap_mean_ms, lime_mean_ms],
    "std_ms" : [shap_std_ms, lime_std_ms],
    "min_ms" : [np.min(shap_times), np.min(lime_times)],
    "max_ms" : [np.max(shap_times), np.max(lime_times)]
})

latency_df.to_csv("results/latency_metrics.csv", index = False)
print("Métricas de latência guardadas em 'results/latency_metrics.csv'")

# ────────────────────────────────────────────────────────────────────────────
#  7. GRÁFICO COMPARATIVO SHAP VS LIME
#     Desenvolvimento de um gráfico para comparar a latência dos dois métodos.
# ────────────────────────────────────────────────────────────────────────────

# Cria a base do gráfico, com tamanho e tema específicos.
fig, ax = plt.subplots(figsize = (8, 6))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

# Define-se os nomes, médias, desvios e cores para os métodos a comparar.
methods = ["SHAP\n(TreeExplainer)", "LIME\n(TabularExplainer)"]
means = [shap_mean_ms, lime_mean_ms]
stds = [shap_std_ms, lime_std_ms]
colors = ["#58a6ff", "#3fb950"]

bars = ax.bar(
    methods,
    means,
    yerr = stds,
    capsize = 6,
    color = colors,
    edgecolor = "none",
    width = 0.45,
    error_kw = {
        "ecolor": "#8b949e",
        "elinewidth": 1.5
    }
)

# Itera sobre as barras para adicionar os valores acima de cada barra, com formatação personalizada.
for bar, val, std in zip(bars, means, stds):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + std + max(means) * 0.02,
        f"{val:.2f} ms",
        ha = "center",
        va = "bottom",
        fontsize = 11,
        fontweight = "bold",
        color = "#e6edf3"
    )

# Calcula o speedup do SHAP em relação ao LIME e adiciona essa informação no gráfico.
speedup = lime_mean_ms / shap_mean_ms
ax.text(
    0.98,
    0.95,
    f"SHAP é {speedup:.0f}x mais rápido",
    transform = ax.transAxes,
    ha = "right",
    va = "top",
    fontsize = 10,
    color = "#ffa657",
    bbox = dict(
        boxstyle = "round,pad = 0.4", 
        facecolor = "#1c2128",
        edgecolor = "#ffa657", 
        alpha = 0.9)
)

# Formata os títulos, cores de texto dos eixos e introduz uma grelha horizontal para melhor visualização dos valores.
ax.set_ylabel(
    "Latência média (ms)", 
    color = "#e6edf3"
)

ax.set_title(
    "Tempo de explicação por pacote: SHAP vs LIME",
    color = "#e6edf3", 
    fontweight = "bold"
)

ax.tick_params(
    colors = "#8b949e"
)

ax.yaxis.grid(
    True, 
    color = "#21262d")

for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

# Ajusta as margens, para que não seja cortado o título ou os valores, e salva o gráfico num ficheiro PNG.
plt.tight_layout()
plt.savefig(
    "results/shap_vs_lime_latency.png", 
    dpi = 150,
    facecolor = fig.get_facecolor())
plt.close()
print("Guardado: results/shap_vs_lime_latency.png")