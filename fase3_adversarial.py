import os, joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

# ──────────────────────────────────────────────────────────────────────────────
#  1. CONFIGURAÇÕES INICIAIS E DIRETÓRIOS
#     Define sementes para reprodutibilidade e garante a pasta de resultados.
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok = True)

RANDOM_STATE = 42
EPS = 0.05 # Magnitude da perturbação (Epsilon).

# Fixa as sementes para garantir que os resultados sejam consistentes em cada execução.
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ──────────────────────────────────────────────────────────────────────────────
#  2. CARREGAMENTO DE DADOS E MODELOS
#     Recupera o modelo XGBoost e os datasets processados na Fase 1.
# ──────────────────────────────────────────────────────────────────────────────

model = joblib.load("models/xgb_baseline.pkl")
top15 = joblib.load("models/top15_features.pkl")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

N_FEATURES = X_train.shape[1]

# ──────────────────────────────────────────────────────────────────────────────
#  3. DEFINIÇÃO DO MODELO SURROGATE (PyTorch)
#     O FGSM precisa de gradientes. Como o XGBoost não os fornece, treinamos 
#     esta rede neuronal (MLP) para imitar o comportamento do classificador.
# ──────────────────────────────────────────────────────────────────────────────

class SurrogateMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

surrogate = SurrogateMLP(N_FEATURES)

# Conversão dos dados para Tensores do PyTorch para treino.
X_train_t = torch.tensor(X_train.values, dtype = torch.float32)
y_train_t = torch.tensor(y_train.values, dtype = torch.long)
X_test_t = torch.tensor(X_test.values, dtype = torch.float32)
y_test_t = torch.tensor(y_test.values, dtype = torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size = 256,
    shuffle = True
)

# ──────────────────────────────────────────────────────────────────────────────
#  4. TREINO DO SURROGATE
#     O objetivo é obter uma acurácia alta (>90%) para garantir a 
#     transferibilidade dos ataques para o modelo original.
# ──────────────────────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(surrogate.parameters(), lr = 0.001)

print("A treinar o surrogate MLP (PyTorch)...")
surrogate.train()
for epoch in range(20):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(surrogate(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Época {epoch+1:02d}/20 — loss: {epoch_loss/len(train_loader):.4f}")

# Avaliação da performance do surrogate no set de teste.
surrogate.eval()
with torch.no_grad():
    logits = surrogate(X_test_t)
    surr_preds = logits.argmax(dim = 1).numpy()

surr_acc = accuracy_score(y_test.values, surr_preds)
print(f"Surrogate accuracy: {surr_acc * 100:.2f}%")

# ──────────────────────────────────────────────────────────────────────────────
#  5. CONFIGURAÇÃO DO ATAQUE ADVERSARIAL (ART + FGSM)
#     Utiliza a biblioteca ART para aplicar o Fast Gradient Sign Method.
# ──────────────────────────────────────────────────────────────────────────────

# Encapsula o modelo PyTorch para ser compatível com a ART.
art_classifier = PyTorchClassifier(
    model = surrogate,
    loss = criterion,
    optimizer = optimizer,
    input_shape = (N_FEATURES,),
    nb_classes = 2,
    clip_values = (0.0, 1.0) # Garante que os valores permanecem no range do Min-Max Scaler.
)

# Configura o FGSM para gerar as perturbações baseadas no gradiente.
fgsm_attack = FastGradientMethod(estimator = art_classifier, eps = EPS)
print(f"FGSM configurado com eps = {EPS}")

# Seleciona apenas os pacotes que são originalmente classificados como "Ataque".
attack_indices = np.where(y_test.values == 1)[0]
X_attacks = X_test.iloc[attack_indices].values.astype(np.float32)

print("A gerar exemplos adversariais com FGSM...")
X_adv = fgsm_attack.generate(x = X_attacks)
X_adv = np.clip(X_adv, 0.0, 1.0)

# ──────────────────────────────────────────────────────────────────────────────
#  6. MÉTRICAS DE EVASÃO E ROBUSTEZ
#     Verifica se o XGBoost (modelo alvo) é enganado ou se mantém a previsão.
# ──────────────────────────────────────────────────────────────────────────────

orig_preds = model.predict(X_attacks)
adv_preds = model.predict(X_adv)

# Identifica ataques que continuam a ser detectados e os que conseguiram evadir.
still_attack = np.where((orig_preds == 1) & (adv_preds == 1))[0]
evaded = np.where((orig_preds == 1) & (adv_preds == 0))[0]
pct_mantidos = len(still_attack) / len(orig_preds) * 100
pct_evasao = len(evaded) / len(orig_preds) * 100

print(f"Ataques mantidos após FGSM: {len(still_attack)}/{len(orig_preds)} ({pct_mantidos:.1f}%)")
print(f"Ataques eludidos (evasão): {len(evaded)}/{len(orig_preds)} ({pct_evasao:.1f}%)")

if len(still_attack) == 0:
    raise ValueError(
        "Nenhum ataque mantido após FGSM. "
        "Reduz EPS para 0.01 ou 0.02 e volta a correr."
    )

# ──────────────────────────────────────────────────────────────────────────────
#  7. ANÁLISE XAI: SHAP REAL VS ADVERSARIAL
#     Compara as explicações para um ataque que manteve a classificação original.
# ──────────────────────────────────────────────────────────────────────────────

shap_explainer = shap.TreeExplainer(model)

# Escolhe o primeiro exemplo onde a classificação "Attack" não mudou.
idx = still_attack[0]
x_real = pd.DataFrame(X_attacks[idx:idx+1], columns = top15)
x_adv = pd.DataFrame(X_adv[idx:idx+1], columns = top15)

shap_real = shap_explainer(x_real)
shap_adv = shap_explainer(x_adv)

# Extrai as top features para verificar se a explicação foi manipulada.
top5_real = (
    pd.Series(np.abs(shap_real.values[0]), index = top15)
    .sort_values(ascending = False)
    .head(5)
)
top5_adv = (
    pd.Series(np.abs(shap_adv.values[0]), index = top15)
    .sort_values(ascending = False)
    .head(5)
)

print("\nTop feature REAL: ", top5_real.index[0])
print("Top feature ADVERSÁRIA: ", top5_adv.index[0])

changed = top5_real.index[0] != top5_adv.index[0]
print(f"\nExplicação mudou? {'SIM ✅' if changed else 'NÃO — tenta aumentar EPS'}")

prob_real = model.predict_proba(x_real)[0][1]
prob_adv = model.predict_proba(x_adv)[0][1]

# ──────────────────────────────────────────────────────────────────────────────
#  8. VISUALIZAÇÃO DOS PAINÉIS SHAP
#     Cria uma comparação visual entre o pacote real e o manipulado.
# ──────────────────────────────────────────────────────────────────────────────

def plot_shap_panel(ax, shap_vals, title, title_color, subtitle):
    """Função auxiliar para desenhar o gráfico de barras SHAP horizontal."""
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
    feats = [top15[i] for i in sorted_idx]
    vals = shap_vals[sorted_idx]
    # Vermelho para valores que empurram para "Attack", azul para o contrário.
    colors = ["#f85149" if v > 0 else "#58a6ff" for v in vals]
    
    ax.barh(
        range(len(feats)), vals[::-1],
        color = colors[::-1], edgecolor = "none", height = 0.6
    )
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f[:30] for f in feats[::-1]], color = "#e6edf3", fontsize = 9)
    ax.set_xlabel("SHAP Value", color = "#8b949e")
    ax.set_title(
        f"{title}\n{subtitle}", 
        color = title_color,
        fontsize = 12, 
        fontweight = "bold"
    )
    ax.set_facecolor("#161b22")
    ax.tick_params(colors = "#8b949e")
    ax.axvline(0, color = "#30363d", linewidth = 1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

fig = plt.figure(figsize = (18, 7), facecolor = "#0d1117")
gs = gridspec.GridSpec(1, 2, figure = fig, wspace = 0.35)

plot_shap_panel(
    fig.add_subplot(gs[0]),
    shap_real.values[0],
    "REAL Attack", "#58a6ff",
    f"Attack ({prob_real*100:.1f}%)"
)
plot_shap_panel(
    fig.add_subplot(gs[1]),
    shap_adv.values[0],
    f"ADVERSARIAL Attack (ε = {EPS})", "#f85149",
    f"Attack ({prob_adv*100:.1f}%) — Manipulated Explanation"
)

fig.suptitle(
    "Manipulation of XAI Explanations via FGSM — SHAP Before vs After",
    color = "#e6edf3", fontsize = 14, fontweight = "bold", y = 1.02
)

plt.savefig(
    "results/shap_real_vs_adversarial.png",
    dpi = 150, facecolor = fig.get_facecolor(), bbox_inches = "tight"
)
plt.close()
print("Guardado: results/shap_real_vs_adversarial.png")

# ──────────────────────────────────────────────────────────────────────────────
#  9. ARMAZENAMENTO DE ARTEFACTOS
#     Exporta os dados para CSV para serem usados no benchmarking da Fase 4.
# ──────────────────────────────────────────────────────────────────────────────

x_real.to_csv("results/x_real_attack.csv", index = False)
x_adv.to_csv("results/x_adv_attack.csv", index = False)

pd.DataFrame({
    "feature": top15,
    "shap_real": shap_real.values[0],
    "shap_adv": shap_adv.values[0],
    "abs_real": np.abs(shap_real.values[0]),
    "abs_adv": np.abs(shap_adv.values[0]),
}).to_csv("results/shap_real_vs_adv_values.csv", index=False)

print("Artefactos guardados em 'results/'")