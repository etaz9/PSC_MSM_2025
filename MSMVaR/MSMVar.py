import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from Calvet_MSM_Gab import *

start_date = '2014-01-04'
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


print("Step 6: Estimating parameters for the MSM...")
niter = 1
temperature = 1.0
stepsize = 1.0
k_bar = 5
data = df['Log_Ret'].values[:, np.newaxis]
data = data[1:]
#print(data)
parameters, LL, niter, output = glo_min(k_bar, data, niter, temperature, stepsize)

# name parameters for later use:
b = parameters[0]
m_0 = parameters[1]
gamma_kbar = parameters[2]
sigma = parameters[3]

pi_t = g_pi_t(m_0,k_bar,data,[b, gamma_kbar, sigma])[-1,:]
A = g_t(k_bar, b, gamma_kbar)

print("Parameters from glo_min for Simulated dataset: ", "\n"
 "kbar = ", k_bar,"\n"
 'b = %.5f' % b,"\n"
 'm_0 = %.5f' % m_0,"\n"
 'gamma_kbar = %.5f' % gamma_kbar,"\n"
 'sigma = %.5f' % sigma,"\n")

def VaR_alpha(returns, vol, alpha):
    z_alpha = norm.ppf(alpha)  # ex : -1.645 pour alpha=5%
    VaR = z_alpha * vol
    violations_bool = returns < VaR  # Série booléenne True/False
    violations_rate = violations_bool.sum() / len(returns)  # Taux global de violations
    return alpha, violations_rate, violations_bool

df["MSM_Vol"] = compute_MSM_vol(data, parameters, k_bar)[0]
Quantiles = [0.01, 0.05, 0.1]


for i in range(3):
    v = VaR_alpha(df["Log_Ret"], df["MSM_Vol"], Quantiles[i])[1]
    print(f"VaR à {100*(1-Quantiles[i]):.0f}% : {v:.2%} de violations")


confidence_levels = [0.90, 0.95, 0.99] 
z_scores = [norm.ppf(1 - alpha) for alpha in confidence_levels]
years = sorted(df.index.year.unique())

print("Dates disponibles :", df.index.min(), "->", df.index.max())
print("Années uniques :", df.index.year.unique())

results = []

for year in years:
    df_year = df[df.index.year == year].copy()
    if df_year.empty:
        continue
    for i in range(3):
        z = z_scores[i]
        VaR = z * df_year["MSM_Vol"]
        violations = (df_year["Log_Ret"] < VaR).sum()
        violation_rate = violations / len(df_year)
        results.append({
            "Année": year,
            "Alpha": int((Quantiles[i]) * 100),
            "Taux de violation (%)": round(violation_rate * 100, 2)
        })

var_table = pd.DataFrame(results)
var_pivot = var_table.pivot(index="Année", columns="Alpha", values="Taux de violation (%)")


# --- Étape 6 : Afficher joliment ---
print("\n📊 Tableau des taux de violation VaR (%)")
print(var_pivot.round(2))

# --- Nouvelle heatmap : agrégation globale ---
global_results = []

for i in range(3):
    z = z_scores[i]
    VaR = z * df["MSM_Vol"]
    #VaR = z * df2["Volatility"]
    violations = (df["Log_Ret"] < VaR).sum()
    violation_rate = violations / len(df)
    global_results.append(round(violation_rate * 100, 2))

# Convertir en DataFrame 1 ligne (index = 'Global')
global_df = pd.DataFrame([global_results], columns=[int((alpha) * 100) for alpha in Quantiles], index=["Période totale"])

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
# plt.savefig("figures/var_violation_subfigures.png", dpi=300, bbox_inches="tight")


alpha_target = 0.05  # exemple pour 1% VaR
alpha, violation_rate, violations = VaR_alpha(df["Log_Ret"], df["MSM_Vol"], alpha_target)


print(f"Taux de violation pour {alpha_target*100:.0f}% VaR : {violation_rate:.2%}")

# Extraire les dates de violation
rupture_dates = df.index[violations]

# Transformer en DataFrame pour export
ruptures_df = pd.DataFrame({'Date': rupture_dates})

# Sauvegarder dans un CSV
ruptures_df.to_csv("ruptures_var.csv", index=False)

print(f"✅ Exporté {len(ruptures_df)} ruptures dans 'ruptures_var.csv'")