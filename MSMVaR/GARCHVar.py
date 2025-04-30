import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import seaborn as sns

# Charger le fichier
start_date = '2014-01-02'
end_date = '2025-03-09'

file_path = "BTC-USD.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Conversion des colonnes
df["Price"] = df["Price"].astype(float)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Mettre la colonne "Date" comme index
df.set_index("Date", inplace=True)

# Filtrer sur la période souhaitée
df = df.loc[start_date:end_date]

# Calcul des log-returns
df["Log_Ret"] = np.log(df["Price"] / df["Price"].shift(1))
df = df.dropna()

# Ajuster un modèle GARCH(1,1)
garch_model = arch_model(df["Log_Ret"] * 100, vol="Garch", p=1, q=1)
garch_fit = garch_model.fit(disp="off")

# Obtenir la volatilité conditionnelle
df["GARCH_Vol"] = garch_fit.conditional_volatility
df["GARCH_Vol"] = df["GARCH_Vol"] / 100
# Tracer la volatilité
def VaR_alpha(returns, vol, alpha):
    z_alpha = norm.ppf(alpha)  # ex : -1.645 pour alpha=5%
    VaR = z_alpha * vol
    violations_bool = returns < VaR  # Série booléenne True/False
    violations_rate = violations_bool.sum() / len(returns)  # Taux global de violations
    return alpha, violations_rate, violations_bool



alpha = 0.05
z_alpha = norm.ppf(alpha)
VaR_series = z_alpha * df["GARCH_Vol"]





alpha = 0.05
z_alpha = norm.ppf(alpha)
VaR_series = z_alpha * df["GARCH_Vol"]

# Filtrer les vraies violations : returns < VaR
violations = df["Log_Ret"] < VaR_series



confidence_levels = [0.90, 0.95, 0.99]
z_scores = {alpha: norm.ppf(1 - alpha) for alpha in confidence_levels}
years = sorted(df.index.year.unique())

print("Dates disponibles :", df.index.min(), "->", df.index.max())
print("Années uniques :", df.index.year.unique())

results = []

for year in years:
    df_year = df[df.index.year == year].copy()
    if df_year.empty:
        continue
    for alpha in confidence_levels:
        z = z_scores[alpha]
        VaR = z * df_year["GARCH_Vol"]
        violations = (df_year["Log_Ret"] < VaR).sum()
        violation_rate = violations / len(df_year)
        results.append({
            "Année": year,
            "Alpha": int(alpha * 100),
            "Taux de violation (%)": round(violation_rate * 100, 2)
        })

var_table = pd.DataFrame(results)
var_pivot = var_table.pivot(index="Année", columns="Alpha", values="Taux de violation (%)")
#
global_results = []

for alpha in confidence_levels:
    z = z_scores[alpha]
    VaR = z * df["GARCH_Vol"]
    #VaR = z * df2["Volatility"]
    violations = (df["Log_Ret"] < VaR).sum()
    violation_rate = violations / len(df)
    global_results.append(round(violation_rate * 100, 2))

# Convertir en DataFrame 1 ligne (index = 'Global')
global_df = pd.DataFrame([global_results], columns=[int( alpha * 100) for alpha in confidence_levels], index=["Période totale"])

# Affichage en console (optionnel)
print("\n📊 Taux de violation global (toutes années confondues)")
print(global_df)

fig, axes = plt.subplots(
    2, 1,
    figsize=(10, 8),
    gridspec_kw={'height_ratios': [3, 0.5]},
    constrained_layout=True
)
# Heatmap 1 : par année
sns.heatmap(var_pivot, ax=axes[0], annot=True, fmt=".2f", cmap="YlGnBu",
            linewidths=0.5, cbar_kws={"label": "Taux de violation (%)"})
axes[0].set_title("Violations de la Value at Risk par Année")
axes[0].set_xlabel("Niveau de confiance (%)")
axes[0].set_ylabel("Année")

# Heatmap 2 : période globale
sns.heatmap(global_df, ax=axes[1], annot=True, fmt=".2f", cmap="YlGnBu",
            linewidths=0.5, cbar_kws={"label": "Taux de violation (%)"})
axes[1].set_title("Violations de la Value at Risk — Période Totale")
axes[1].set_xlabel("Niveau de confiance (%)")
axes[1].set_ylabel("")

# Sauvegarder ou afficher
plt.suptitle("Taux de violation de la VaR par modèle MSM", fontsize=14, y=1.02)
plt.show()

alpha_target = 0.08  # exemple pour 1% VaR
alpha, violation_rate, violations = VaR_alpha(df["Log_Ret"], df["GARCH_Vol"], alpha_target)

print(f"Taux de violation pour {alpha_target*100:.0f}% VaR : {violation_rate:.2%}")

# Extraire les dates de violation
rupture_dates = df.index[violations]

# Transformer en DataFrame pour export
ruptures_df = pd.DataFrame({'Date': rupture_dates})

# Sauvegarder dans un CSV
ruptures_df.to_csv("ruptures_var.csv", index=False)

print(f"✅ Exporté {len(ruptures_df)} ruptures dans 'ruptures_var.csv'")