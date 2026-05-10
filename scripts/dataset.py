import pandas as pd
import random
import io


def extracao_bruta_multiclasse(input_path, output_path, target_total=200053):
    print(f"🚜 Iniciando Extração Bruta (Modo Sobrevivência): {input_path}")

    target_col = "Attack_type"
    limite_por_classe = 13337
    repositorio = {}
    colunas_finais = []

    # 1. Identificar o cabeçalho
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")
        if target_col not in header:
            # Tenta encontrar a coluna caso o nome varie
            for h in header:
                if "attack" in h.lower() or "type" in h.lower():
                    target_col = h
                    break
        target_idx = header.index(target_col)
        print(f"📍 Coluna alvo encontrada no índice {target_idx}: {target_col}")

    # 2. Percorrer o arquivo linha por linha (ignorando erros de parser)
    print("⛏️ Minerando... Este processo vai ler o arquivo até o fim.")

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        next(f)  # Pula o cabeçalho
        for i, line in enumerate(f):
            parts = line.strip().split(",")

            # Garante que a linha tem o número correto de colunas
            if len(parts) == len(header):
                cls = parts[target_idx].strip()

                if cls not in repositorio:
                    repositorio[cls] = []

                if len(repositorio[cls]) < limite_por_classe:
                    # Guardamos a linha inteira como uma lista para processar depois
                    repositorio[cls].append(parts)

            if i % 1000000 == 0 and i > 0:
                print(f"📡 {i//1000000} Milhões de linhas lidas...")
                encontradas = {k: len(v) for k, v in repositorio.items() if len(v) > 0}
                print(f"   Classes atuais: {encontradas}")

    # 3. Processar e Converter para Numérico
    print("\n⚖️ Consolidando e limpando dados...")
    dados_finais = []
    for cls, linhas in repositorio.items():
        print(f" - {cls}: {len(linhas)} amostras")
        for p in linhas:
            d = {}
            for idx, val in enumerate(p):
                col_name = header[idx]
                if col_name == target_col:
                    d[col_name] = val
                else:
                    # Tenta converter para float, se falhar coloca 0.0
                    try:
                        d[col_name] = float(val)
                    except:
                        d[col_name] = 0.0
            dados_finais.append(d)

    # 4. Criar DataFrame e Salvar
    if not dados_finais:
        print("❌ Nenhuma linha válida foi extraída!")
        return

    df = pd.DataFrame(dados_finais)

    # Remover colunas que ficaram só com zeros (lixo/texto)
    df = df.loc[:, (df != 0).any(axis=0)]

    # Ajuste final e Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if len(df) > target_total:
        df = df.head(target_total)

    df.to_csv(output_path, index=False)
    print(f"\n✨ SUCESSO! Dataset final gerado: {output_path} ({df.shape})")


if __name__ == "__main__":
    extracao_bruta_multiclasse(
        "./db/'ML-EdgeIIoT-dataset.csv", "./db/ML_EdgeIIoT_BALANCEADO_FINAL.csv"
    )
