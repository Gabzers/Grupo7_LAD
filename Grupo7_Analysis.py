import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Import os for file existence check
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuração do estilo dos gráficos
sns.set(style="whitegrid")

# Carregar os dados com validação
def load_csv(file_path):
    """Carrega um arquivo CSV para um DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please provide a valid path.")
    return pd.read_csv(file_path)

# Mostrar informações básicas
def show_basic_info(df):
    """Exibe informações básicas sobre o dataset."""
    print("\nInformações básicas:")
    print(df.info())
    print("\nPrimeiras linhas do DataFrame:")
    print(df.head())

# Estatísticas descritivas
def describe_dataset(df):
    """Mostra estatísticas descritivas do dataset."""
    print("\nEstatísticas Descritivas:")
    print(df.describe(include='all'))
    print("\nValores Nulos por Coluna:")
    print(df.isnull().sum())

# Análise gráfica
def graphical_analysis(df):
    """Realiza a análise gráfica com 5 visualizações principais."""
    # 1. Distribuição dos tipos de ataque
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y="Attack_type", order=df["Attack_type"].value_counts().index, palette="coolwarm")
    plt.title("Distribuição dos Tipos de Ataque")
    plt.xlabel("Número de Ocorrências")
    plt.ylabel("Tipo de Ataque")
    plt.show()
    
    # 2. Heatmap de correlação entre features numéricas
    plt.figure(figsize=(10, 8))
    selected_features = ["flow_duration", "fwd_pkts_tot", "bwd_pkts_tot", "flow_pkts_per_sec",
                         "payload_bytes_per_second", "idle.avg", "active.avg"]
    corr_matrix = df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlação entre Features Selecionadas")
    plt.show()
    
    # 3. Distribuição dos pacotes por protocolo
    plt.figure(figsize=(12, 6))
    df["proto"].value_counts().head(10).plot(kind="pie", autopct="%1.1f%%", cmap="tab10", startangle=90)
    plt.title("Distribuição dos Protocolos de Rede (Top 10)")
    plt.ylabel("")
    plt.show()
    
    # 4. Distribuição do fluxo de duração
    plt.figure(figsize=(12, 6))
    sns.histplot(df["flow_duration"], bins=50, kde=True, color="blue")
    plt.title("Distribuição da Duração do Fluxo de Rede")
    plt.xlabel("Duração do Fluxo")
    plt.ylabel("Frequência")
    plt.show()
    
    # 5. Boxplot do tamanho de pacotes enviados
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Attack_type", y="fwd_pkts_payload.tot", palette="coolwarm")
    plt.ylim(0, df["fwd_pkts_payload.tot"].quantile(0.99))
    plt.title("Distribuição do Tamanho de Pacotes Enviados por Tipo de Ataque")
    plt.xticks(rotation=90)
    plt.show()

# Additional graphical analyses
def additional_graphical_analysis(df):
    """Realiza análises gráficas adicionais baseadas no dataset."""
    # 1. Distribuição de valores nulos por coluna (ajustada para evitar sobrecarga visual)
    null_counts = df.isnull().sum().sort_values(ascending=False)
    top_nulls = null_counts[null_counts > 0].head(10)  # Mostrar apenas as 10 colunas com mais valores nulos
    others_count = null_counts[null_counts > 0][10:].sum()  # Agregar o restante em "Outros"
    
    if others_count > 0:
        top_nulls["Outros"] = others_count

    plt.figure(figsize=(12, 6))
    top_nulls.plot(kind="bar", color="orange")
    plt.title("Distribuição de Valores Nulos por Coluna (Top 10)")
    plt.xlabel("Colunas")
    plt.ylabel("Número de Valores Nulos")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 2. Top 10 valores únicos em 'proto' (ajustada para evitar sobrecarga visual)
    proto_counts = df["proto"].value_counts()
    top_protos = proto_counts.head(10)  # Mostrar apenas os 10 protocolos mais frequentes
    others_proto_count = proto_counts[10:].sum()  # Agregar o restante em "Outros"
    
    if others_proto_count > 0:
        top_protos["Outros"] = others_proto_count

    plt.figure(figsize=(12, 6))
    top_protos.plot(kind="bar", color="green")
    plt.title("Top 10 Protocolos de Rede (Com Outros Agregados)")
    plt.xlabel("Protocolo")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 3. Relação entre 'flow_duration' e 'fwd_pkts_tot' (ajustada para evitar sobreposição)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="flow_duration", y="fwd_pkts_tot", hue="Attack_type", palette="coolwarm", alpha=0.5)
    plt.title("Relação entre Duração do Fluxo e Pacotes Enviados")
    plt.xlabel("Duração do Fluxo")
    plt.ylabel("Pacotes Enviados (Total)")
    plt.legend(title="Tipo de Ataque", bbox_to_anchor=(1.05, 1), loc="upper left")  # Ajustar a legenda
    plt.tight_layout()
    plt.show()

    # 4. Boxplot de 'payload_bytes_per_second' por 'Attack_type' (ajustada para evitar outliers extremos)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Attack_type", y="payload_bytes_per_second", palette="viridis", showfliers=False)
    plt.title("Distribuição de Payload por Tipo de Ataque (Sem Outliers)")
    plt.xlabel("Tipo de Ataque")
    plt.ylabel("Payload (Bytes por Segundo)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 5. Distribuição de 'idle.avg' com KDE (ajustada para evitar sobrecarga visual)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["idle.avg"], shade=True, color="purple", bw_adjust=0.5)  # Ajustar suavização
    plt.title("Distribuição de Tempo Ocioso Médio")
    plt.xlabel("Idle (Média)")
    plt.ylabel("Densidade")
    plt.tight_layout()
    plt.show()

# Perform statistical analysis
def perform_statistical_analysis(df):
    numeric_data = df.select_dtypes(include=[np.number])  # Select only numeric columns
    stats = numeric_data.describe().T  # Transpose for better visualization
    stats['range'] = stats['max'] - stats['min']  # Add range calculation
    stats['variance'] = numeric_data.var()  # Variance
    stats['mode'] = numeric_data.mode().iloc[0]  # Add mode calculation
    return stats

# Additional visualizations
def analyze_distribution(df, column, title, xlabel, ylabel, bins=30, color='blue'):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=bins, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def analyze_correlation(df, selected_features):
    plt.figure(figsize=(10, 8))
    corr_matrix = df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

# Tkinter GUI for analysis
def show_summary_statistics(df):
    """Exibe estatísticas descritivas com explicações intuitivas."""
    stats = perform_statistical_analysis(df)
    stats = stats.round(1)  # Round all numeric values to one decimal place
    stats.insert(0, "Variable", stats.index)  # Add a "Variable" column for clarity
    stats_window = tk.Toplevel(root)
    stats_window.title("Summary Statistics")
    stats_window.geometry("900x600")  # Adjusted width for the new column

    # Add explanations for each metric
    explanations = {
        "Variable": "Nome da variável (coluna do dataset)",
        "count": "Número de valores não nulos",
        "mean": "Média (valor médio)",
        "std": "Desvio padrão (dispersão dos dados)",
        "min": "Valor mínimo",
        "25%": "Primeiro quartil (25% dos dados abaixo deste valor)",
        "50%": "Mediana (50% dos dados abaixo deste valor)",
        "75%": "Terceiro quartil (75% dos dados abaixo deste valor)",
        "max": "Valor máximo",
        "range": "Intervalo (máximo - mínimo)",
        "variance": "Variância (dispersão dos dados)",
        "mode": "Moda (valor mais frequente)"
    }

    # Create a frame for explanations
    explanation_frame = tk.Frame(stats_window)
    explanation_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    tk.Label(explanation_frame, text="Explicações das Estatísticas:", font=("Arial", 12, "bold")).pack(anchor="w")
    for metric, explanation in explanations.items():
        tk.Label(explanation_frame, text=f"{metric}: {explanation}", font=("Arial", 10)).pack(anchor="w")

    # Create a Treeview for the statistics
    tree_frame = tk.Frame(stats_window)
    tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    tree = ttk.Treeview(tree_frame, columns=list(stats.columns), show='headings')
    tree.pack(fill=tk.BOTH, expand=True)

    for col in stats.columns:
        tree.heading(col, text=col, anchor="center")
        tree.column(col, width=120, anchor="center")  # Adjusted width for better readability

    for _, row in stats.iterrows():
        tree.insert("", tk.END, values=list(row))

def show_distribution(df, column, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30, color='blue', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plot_window = tk.Toplevel(root)
    plot_window.title(title)
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Functions for individual graphical analyses
def show_attack_type_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y="Attack_type", order=df["Attack_type"].value_counts().index, palette="coolwarm")
    plt.title("Distribuição dos Tipos de Ataque")
    plt.xlabel("Número de Ocorrências")
    plt.ylabel("Tipo de Ataque")
    plt.show()

def show_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    selected_features = ["flow_duration", "fwd_pkts_tot", "bwd_pkts_tot", "flow_pkts_per_sec",
                         "payload_bytes_per_second", "idle.avg", "active.avg"]
    corr_matrix = df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlação entre Features Selecionadas")
    plt.show()

def show_protocol_distribution(df):
    plt.figure(figsize=(12, 6))
    df["proto"].value_counts().head(10).plot(kind="pie", autopct="%1.1f%%", cmap="tab10", startangle=90)
    plt.title("Distribuição dos Protocolos de Rede (Top 10)")
    plt.ylabel("")
    plt.show()

def show_flow_duration_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df["flow_duration"], bins=50, kde=True, color="blue")
    plt.title("Distribuição da Duração do Fluxo de Rede")
    plt.xlabel("Duração do Fluxo")
    plt.ylabel("Frequência")
    plt.show()

def show_packet_size_boxplot(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Attack_type", y="fwd_pkts_payload.tot", palette="coolwarm")
    plt.ylim(0, df["fwd_pkts_payload.tot"].quantile(0.99))
    plt.title("Distribuição do Tamanho de Pacotes Enviados por Tipo de Ataque")
    plt.xticks(rotation=90)
    plt.show()

def show_null_values_distribution(df):
    plt.figure(figsize=(12, 6))
    df.isnull().sum().sort_values(ascending=False).plot(kind="bar", color="orange")
    plt.title("Distribuição de Valores Nulos por Coluna")
    plt.xlabel("Colunas")
    plt.ylabel("Número de Valores Nulos")
    plt.xticks(rotation=45)
    plt.show()

def show_active_avg_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["active.avg"], shade=True, color="red")
    plt.title("Distribuição de Tempo Ativo Médio")
    plt.xlabel("Active (Média)")
    plt.ylabel("Densidade")
    plt.show()

def show_idle_avg_boxplot(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="proto", y="idle.avg", palette="Set2")
    plt.ylim(0, df["idle.avg"].quantile(0.99))
    plt.title("Distribuição de Tempo Ocioso por Protocolo")
    plt.xlabel("Protocolo")
    plt.ylabel("Idle (Média)")
    plt.xticks(rotation=45)
    plt.show()

def show_packet_vs_payload_scatter(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="flow_pkts_per_sec", y="payload_bytes_per_second", hue="Attack_type", palette="coolwarm", alpha=0.7)
    plt.title("Relação entre Pacotes por Segundo e Payload por Segundo")
    plt.xlabel("Pacotes por Segundo")
    plt.ylabel("Payload (Bytes por Segundo)")
    plt.legend(title="Tipo de Ataque")
    plt.show()

def show_bwd_pkts_vs_flow_duration(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="bwd_pkts_tot", y="flow_duration", hue="Attack_type", palette="coolwarm", alpha=0.7)
    plt.title("Relação entre Pacotes Recebidos e Duração do Fluxo")
    plt.xlabel("Pacotes Recebidos (Total)")
    plt.ylabel("Duração do Fluxo")
    plt.legend(title="Tipo de Ataque")
    plt.show()

# Main Tkinter window
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "RT_IOT2022.csv")
    df = load_csv(file_path)

    root = tk.Tk()
    root.title("Statistical Analysis Interface")
    root.geometry("400x800")

    tk.Label(root, text="Statistical Analysis Dashboard", font=("Arial", 16)).pack(pady=10)

    # Buttons for analysis
    tk.Button(root, text="Summary Statistics", command=lambda: show_summary_statistics(df)).pack(pady=5)
    tk.Button(root, text="Distribution of Flow Duration", command=lambda: show_distribution(df, 'flow_duration', 'Flow Duration Distribution', 'Flow Duration', 'Frequency')).pack(pady=5)

    # Buttons for individual graphical analyses
    tk.Button(root, text="Distribuição dos Tipos de Ataque", command=lambda: show_attack_type_distribution(df)).pack(pady=5)
    tk.Button(root, text="Heatmap de Correlação", command=lambda: show_correlation_heatmap(df)).pack(pady=5)
    tk.Button(root, text="Distribuição de Protocolos", command=lambda: show_protocol_distribution(df)).pack(pady=5)
    tk.Button(root, text="Distribuição da Duração do Fluxo", command=lambda: show_flow_duration_distribution(df)).pack(pady=5)
    tk.Button(root, text="Boxplot do Tamanho de Pacotes", command=lambda: show_packet_size_boxplot(df)).pack(pady=5)
    tk.Button(root, text="Distribuição de Valores Nulos", command=lambda: show_null_values_distribution(df)).pack(pady=5)
    tk.Button(root, text="Distribuição de Tempo Ativo Médio", command=lambda: show_active_avg_distribution(df)).pack(pady=5)
    tk.Button(root, text="Boxplot de Tempo Ocioso", command=lambda: show_idle_avg_boxplot(df)).pack(pady=5)
    tk.Button(root, text="Relação Pacotes vs Payload", command=lambda: show_packet_vs_payload_scatter(df)).pack(pady=5)
    tk.Button(root, text="Relação Pacotes Recebidos vs Duração", command=lambda: show_bwd_pkts_vs_flow_duration(df)).pack(pady=5)

    root.mainloop()
