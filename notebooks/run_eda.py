import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nbformat as nbf

# Rutas relativas al propio archivo: el script debe correr desde cualquier
# directorio y en cualquier máquina.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
artifacts_dir = os.path.join(BASE_DIR, 'notebooks', 'eda_plots')
os.makedirs(artifacts_dir, exist_ok=True)
data_dir = os.path.join(BASE_DIR, 'data')

# Load data
df_clientes = pd.read_csv(f'{data_dir}/dataset_clientes.csv')
df_ofertas = pd.read_csv(f'{data_dir}/catalogo_ofertas_entrega.csv')
df_campanias = pd.read_csv(f'{data_dir}/historial_campanias.csv')

findings = []
findings.append("# EDA Findings")

# 1. MT Eligible vs Non-eligible
mt_elegible_counts = df_clientes['elegible_mt'].value_counts(normalize=True) * 100
findings.append(f"\n## Movistar Total Eligibility")
findings.append(f"- Eligible for MT: {mt_elegible_counts.get(True, 0):.2f}%")
findings.append(f"- Not Eligible: {mt_elegible_counts.get(False, 0):.2f}%")

plt.figure(figsize=(6, 4))
sns.countplot(data=df_clientes, x='elegible_mt')
plt.title('Distribución de Clientes Elegibles a MT')
plt.savefig(f'{artifacts_dir}/mt_elegible_dist.png')
plt.close()

# 2. Acceptance Rates (Filtering out 'pendiente' as agreed)
df_campanias_dec = df_campanias[df_campanias['resultado'] != 'pendiente'].copy()
df_campanias_dec['aceptada'] = (df_campanias_dec['resultado'] == 'aceptada').astype(int)

# Overall acceptance rate
overall_acc = df_campanias_dec['aceptada'].mean() * 100
findings.append(f"\n## Acceptance Rates (excluding 'pendiente')")
findings.append(f"- Overall Acceptance Rate: {overall_acc:.2f}%")

# Acceptance by Channel
acc_by_channel = df_campanias_dec.groupby('canal')['aceptada'].agg(['mean', 'count']).sort_values('mean', ascending=False)
acc_by_channel['mean'] = acc_by_channel['mean'] * 100
findings.append(f"\n### By Channel")
for idx, row in acc_by_channel.iterrows():
    findings.append(f"- {idx}: {row['mean']:.2f}% (N={row['count']})")

plt.figure(figsize=(8, 5))
sns.barplot(x=acc_by_channel.index, y=acc_by_channel['mean'])
plt.title('Tasa de Aceptación por Canal (%)')
plt.ylabel('Aceptación (%)')
plt.savefig(f'{artifacts_dir}/acc_by_channel.png')
plt.close()

# Acceptance by Offer Type
acc_by_offer_type = df_campanias_dec.groupby('tipo_oferta')['aceptada'].agg(['mean', 'count']).sort_values('mean', ascending=False)
acc_by_offer_type['mean'] = acc_by_offer_type['mean'] * 100
findings.append(f"\n### By Offer Type")
for idx, row in acc_by_offer_type.iterrows():
    findings.append(f"- {idx}: {row['mean']:.2f}% (N={row['count']})")

plt.figure(figsize=(10, 5))
sns.barplot(x=acc_by_offer_type.index, y=acc_by_offer_type['mean'])
plt.title('Tasa de Aceptación por Tipo de Oferta (%)')
plt.ylabel('Aceptación (%)')
plt.savefig(f'{artifacts_dir}/acc_by_offer_type.png')
plt.close()

# Acceptance for MT variants specifically
df_mt_offers = df_campanias_dec[df_campanias_dec['oferta_es_mt'] == True]
if not df_mt_offers.empty:
    mt_acc = df_mt_offers['aceptada'].mean() * 100
    findings.append(f"\n### MT Offers Acceptance")
    findings.append(f"- MT Offers Acceptance Rate: {mt_acc:.2f}% (N={len(df_mt_offers)})")
    
    # MT Acceptance by Eligibility
    acc_mt_elig = df_mt_offers.groupby('elegible_mt')['aceptada'].mean() * 100
    findings.append(f"- Acceptance if Eligible: {acc_mt_elig.get(True, 0):.2f}%")
    findings.append(f"- Acceptance if NOT Eligible: {acc_mt_elig.get(False, 0):.2f}%")

# 3. Consumption vs Acceptance
# Merge campaigns with client consumption
df_merged = df_campanias_dec.merge(df_clientes[['cliente_id', 'consumo_datos_gb_prom', 'monto_facturado_prom']], on='cliente_id', how='left')

findings.append(f"\n## Consumption & Spend vs Acceptance")
mean_gb_acc = df_merged[df_merged['aceptada']==1]['consumo_datos_gb_prom'].mean()
mean_gb_rej = df_merged[df_merged['aceptada']==0]['consumo_datos_gb_prom'].mean()
findings.append(f"- Avg Data (GB) - Accepted: {mean_gb_acc:.2f} vs Rejected: {mean_gb_rej:.2f}")

mean_spend_acc = df_merged[df_merged['aceptada']==1]['monto_facturado_prom'].mean()
mean_spend_rej = df_merged[df_merged['aceptada']==0]['monto_facturado_prom'].mean()
findings.append(f"- Avg Spend (Soles) - Accepted: {mean_spend_acc:.2f} vs Rejected: {mean_spend_rej:.2f}")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_merged, x='aceptada', y='consumo_datos_gb_prom', showfliers=False)
plt.title('Consumo de Datos (GB) vs Aceptación')
plt.savefig(f'{artifacts_dir}/consumo_vs_acc.png')
plt.close()

# Save findings to a markdown file
with open(f'{artifacts_dir}/eda_summary.md', 'w') as f:
    f.write('\n'.join(findings))

# Print findings to stdout
print('\n'.join(findings))

# Create a Jupyter Notebook for the repository
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# Exploratory Data Analysis (EDA)\nAI Telecom Challenge 2026 NBO"),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns"),
    nbf.v4.new_code_cell("df_clientes = pd.read_csv('../data/dataset_clientes.csv')\ndf_ofertas = pd.read_csv('../data/catalogo_ofertas_entrega.csv')\ndf_campanias = pd.read_csv('../data/historial_campanias.csv')"),
    nbf.v4.new_markdown_cell("## 1. Movistar Total Eligibility"),
    nbf.v4.new_code_cell("df_clientes['elegible_mt'].value_counts(normalize=True).plot(kind='bar')\nplt.title('MT Eligibility')\nplt.show()"),
    nbf.v4.new_markdown_cell("## 2. Acceptance Rates"),
    nbf.v4.new_code_cell("df_valid = df_campanias[df_campanias['resultado'] != 'pendiente'].copy()\ndf_valid['aceptada'] = (df_valid['resultado'] == 'aceptada').astype(int)\ndf_valid.groupby('canal')['aceptada'].mean().sort_values().plot(kind='barh')\nplt.title('Acceptance by Channel')\nplt.show()"),
]
nb['cells'] = cells
with open(os.path.join(BASE_DIR, 'notebooks', '01_eda.ipynb'), 'w') as f:
    nbf.write(nb, f)
print("EDA Notebook 01_eda.ipynb created successfully.")
