import pandas as pd

# Carrega o dataset que acabamos de gerar
df = pd.read_csv("./ML_EdgeIIoT_SMOTE.csv")

print("=== Verificação de Balanceamento ===")
print(f"Total de linhas: {len(df)}")
print("\nContagem por Classe:")
print(df["Attack_type"].value_counts())

print("\nProporção (%) por Classe:")
print(df["Attack_type"].value_counts(normalize=True) * 100)
