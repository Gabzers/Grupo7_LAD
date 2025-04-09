import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Import os for file existence check
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure the style of the plots
sns.set(style="whitegrid")

# Load data with validation
def load_csv(file_path):
    """Loads a CSV file into a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please provide a valid path.")
    return pd.read_csv(file_path)

# Show basic information
def show_basic_info(df):
    """Displays basic information about the dataset."""
    print("\nBasic Information:")
    print(df.info())
    print("\nFirst rows of the DataFrame:")
    print(df.head())

# Descriptive statistics
def describe_dataset(df):
    """Displays descriptive statistics of the dataset."""
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))
    print("\nNull Values by Column:")
    print(df.isnull().sum())

# Graphical analysis
def graphical_analysis(df):
    """Performs graphical analysis with 5 main visualizations."""
    # 1. Distribution of attack types
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y="Attack_type", order=df["Attack_type"].value_counts().index, palette="coolwarm")
    plt.title("Distribution of Attack Types")
    plt.xlabel("Number of Occurrences")
    plt.ylabel("Attack Type")
    plt.show()
    
    # 2. Correlation heatmap for numerical features
    plt.figure(figsize=(10, 8))
    selected_features = ["flow_duration", "fwd_pkts_tot", "bwd_pkts_tot", "flow_pkts_per_sec",
                         "payload_bytes_per_second", "idle.avg", "active.avg"]
    corr_matrix = df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix for Selected Features")
    plt.show()
    
    # 3. Distribution of packets by protocol
    plt.figure(figsize=(12, 6))
    df["proto"].value_counts().head(10).plot(kind="pie", autopct="%1.1f%%", cmap="tab10", startangle=90)
    plt.title("Distribution of Network Protocols (Top 10)")
    plt.ylabel("")
    plt.show()
    
    # 4. Distribution of flow duration
    plt.figure(figsize=(12, 6))
    sns.histplot(df["flow_duration"], bins=50, kde=True, color="blue")
    plt.title("Distribution of Network Flow Duration")
    plt.xlabel("Flow Duration")
    plt.ylabel("Frequency")
    plt.show()
    
    # 5. Boxplot of packet sizes sent
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Attack_type", y="fwd_pkts_payload.tot", palette="coolwarm")
    plt.ylim(0, df["fwd_pkts_payload.tot"].quantile(0.99))
    plt.title("Distribution of Packet Sizes Sent by Attack Type")
    plt.xticks(rotation=90)
    plt.show()

# Additional graphical analyses
def additional_graphical_analysis(df):
    """Performs additional graphical analyses based on the dataset."""
    # 1. Distribution of null values by column (adjusted to avoid visual overload)
    null_counts = df.isnull().sum().sort_values(ascending=False)
    top_nulls = null_counts[null_counts > 0].head(10)  # Show only the top 10 columns with the most null values
    others_count = null_counts[null_counts > 0][10:].sum()  # Aggregate the rest as "Others"
    
    if others_count > 0:
        top_nulls["Others"] = others_count

    plt.figure(figsize=(12, 6))
    top_nulls.plot(kind="bar", color="orange")
    plt.title("Distribution of Null Values by Column (Top 10)")
    plt.xlabel("Columns")
    plt.ylabel("Number of Null Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 2. Top 10 unique values in 'proto' (adjusted to avoid visual overload)
    proto_counts = df["proto"].value_counts()
    top_protos = proto_counts.head(10)  # Show only the top 10 most frequent protocols
    others_proto_count = proto_counts[10:].sum()  # Aggregate the rest as "Others"
    
    if others_proto_count > 0:
        top_protos["Others"] = others_proto_count

    plt.figure(figsize=(12, 6))
    top_protos.plot(kind="bar", color="green")
    plt.title("Top 10 Network Protocols (With Others Aggregated)")
    plt.xlabel("Protocol")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 3. Relationship between 'flow_duration' and 'fwd_pkts_tot' (adjusted to avoid overlap)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="flow_duration", y="fwd_pkts_tot", hue="Attack_type", palette="coolwarm", alpha=0.5)
    plt.title("Relationship Between Flow Duration and Packets Sent")
    plt.xlabel("Flow Duration")
    plt.ylabel("Packets Sent (Total)")
    plt.legend(title="Attack Type", bbox_to_anchor=(1.05, 1), loc="upper left")  # Adjust legend
    plt.tight_layout()
    plt.show()

    # 4. Boxplot of 'payload_bytes_per_second' by 'Attack_type' (adjusted to avoid extreme outliers)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Attack_type", y="payload_bytes_per_second", palette="viridis", showfliers=False)
    plt.title("Payload Distribution by Attack Type (Without Outliers)")
    plt.xlabel("Attack Type")
    plt.ylabel("Payload (Bytes per Second)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 5. Distribution of 'idle.avg' with KDE (adjusted to avoid visual overload)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["idle.avg"], shade=True, color="purple", bw_adjust=0.5)  # Adjust smoothing
    plt.title("Distribution of Average Idle Time")
    plt.xlabel("Idle (Average)")
    plt.ylabel("Density")
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
    if column == 'flow_duration':  # Apply x-axis range only for flow_duration
        plt.xlim(0, 800)  # Set the range from -10 to 800
    plt.show()

def analyze_correlation(df, selected_features):
    plt.figure(figsize=(10, 8))
    corr_matrix = df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

# Tkinter GUI for analysis
def show_summary_statistics(df):
    """Displays descriptive statistics with intuitive explanations."""
    stats = perform_statistical_analysis(df)
    stats = stats.round(1)  # Round all numeric values to one decimal place
    stats.insert(0, "Variable", stats.index)  # Add a "Variable" column for clarity
    stats_window = tk.Toplevel(root)
    stats_window.title("Summary Statistics")
    stats_window.geometry("900x600")  # Adjusted width for the new column

    # Add explanations for each metric
    explanations = {
        "Variable": "Name of the variable (dataset column)",
        "count": "Number of non-null values",
        "mean": "Mean (average value)",
        "std": "Standard deviation (data dispersion)",
        "min": "Minimum value",
        "25%": "First quartile (25% of data below this value)",
        "50%": "Median (50% of data below this value)",
        "75%": "Third quartile (75% of data below this value)",
        "max": "Maximum value",
        "range": "Range (max - min)",
        "variance": "Variance (data dispersion)",
        "mode": "Mode (most frequent value)"
    }

    # Create a frame for explanations
    explanation_frame = tk.Frame(stats_window)
    explanation_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    tk.Label(explanation_frame, text="Explanations of Statistics:", font=("Arial", 12, "bold")).pack(anchor="w")
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
    plt.title("Distribution of Attack Types")
    plt.xlabel("Number of Occurrences")
    plt.ylabel("Attack Type")
    plt.show()

def show_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    selected_features = ["flow_duration", "fwd_pkts_tot", "bwd_pkts_tot", "flow_pkts_per_sec",
                         "payload_bytes_per_second", "idle.avg", "active.avg"]
    corr_matrix = df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix for Selected Features")
    plt.show()

def show_protocol_distribution(df):
    plt.figure(figsize=(12, 6))
    df["proto"].value_counts().head(10).plot(kind="pie", autopct="%1.1%%", cmap="tab10", startangle=90)
    plt.title("Distribution of Network Protocols (Top 10)")
    plt.ylabel("")
    plt.show()

def show_packet_size_barplots(df):
    """Displays individual bar plots for each variable related to packet sizes."""
    variables = ['fwd_pkts_payload.tot', 'bwd_pkts_payload.tot', 'fwd_pkts_tot', 'bwd_pkts_tot']
    for var in variables:
        plt.figure(figsize=(12, 6))
        avg_packet_size = df.groupby("Attack_type")[var].mean().sort_values(ascending=False)
        avg_packet_size.plot(kind="bar", color="skyblue")
        plt.title(f"Average {var.replace('_', ' ').title()} by Attack Type")
        plt.xlabel("Attack Type")
        plt.ylabel(var.replace('_', ' ').title())
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

def show_active_avg_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["active.avg"], shade=True, color="red")
    plt.title("Distribution of Average Active Time")
    plt.xlabel("Active (Average)")
    plt.ylabel("Density")
    plt.show()

def show_idle_avg_barplot(df):
    """Displays a bar plot for the average idle time by attack type."""
    plt.figure(figsize=(12, 6))
    idle_avg_by_attack = df.groupby("Attack_type")["idle.avg"].mean().sort_values(ascending=False)
    idle_avg_by_attack.plot(kind="bar", color="skyblue")
    plt.title("Average Idle Time by Attack Type")
    plt.xlabel("Attack Type")
    plt.ylabel("Idle Time (Average)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def show_packet_vs_payload_scatter(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="flow_pkts_per_sec", y="payload_bytes_per_second", hue="Attack_type", palette="coolwarm", alpha=0.7)
    plt.title("Relationship Between Packets per Second and Payload per Second")
    plt.xlabel("Packets per Second")
    plt.ylabel("Payload (Bytes per Second)")
    plt.legend(title="Attack Type")
    plt.show()

def show_bwd_pkts_vs_flow_duration(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="bwd_pkts_tot", y="flow_duration", hue="Attack_type", palette="coolwarm", alpha=0.7)
    plt.title("Relationship Between Received Packets and Flow Duration")
    plt.xlabel("Received Packets (Total)")
    plt.ylabel("Flow Duration")
    plt.legend(title="Attack Type")
    plt.show()

# Add down_up_ratio calculation
def calculate_down_up_ratio(df):
    """Calculates the down/up ratio for the dataset."""
    df['down_up_ratio'] = df['bwd_pkts_tot'] / (df['fwd_pkts_tot'] + 1e-9)  # Avoid division by zero
    return df

# Add analysis for flags
def analyze_flags(df):
    """Analyzes the distribution of flags in the dataset."""
    flag_columns = ['flow_SYN_flag_count', 'flow_RST_flag_count', 'flow_FIN_flag_count', 
                    'fwd_PSH_flag_count', 'bwd_PSH_flag_count', 'flow_ACK_flag_count', 
                    'fwd_URG_flag_count', 'bwd_URG_flag_count', 'flow_CWR_flag_count', 
                    'flow_ECE_flag_count']
    flag_sums = df[flag_columns].sum()
    plt.figure(figsize=(12, 6))
    flag_sums.plot(kind="bar", color="teal")
    plt.title("Distribution of Flags")
    plt.xlabel("Flags")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Add analysis for IAT
def analyze_iat(df):
    """Analyzes the distribution of Inter-Arrival Times (IAT)."""
    iat_columns = ['fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.avg', 'bwd_iat.std']
    for col in iat_columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df[col], shade=True, bw_adjust=0.5)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        if col == 'fwd_iat.avg':  # Apply range adjustment only for fwd_iat.avg
            plt.xlim(0, 0.25 * 1e8)
        plt.tight_layout()
        plt.show()

# Add analysis for activity and idle times
def analyze_activity_idle(df):
    """Analyzes the activity and idle times in the dataset."""
    activity_columns = ['active.avg', 'active.tot', 'idle.avg', 'idle.tot']
    for col in activity_columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df[col], shade=True, bw_adjust=0.5)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

# Add analysis for down/up ratio
def analyze_down_up_ratio(df):
    """Analyzes the distribution of the down/up ratio."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['down_up_ratio'], kde=True, bins=30, color='purple')
    plt.title("Distribution of Down/Up Ratio")
    plt.xlabel("Down/Up Ratio")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def show_dataset_characteristics():
    """Displays the characteristics of the dataset in a maximizable window."""
    characteristics = (
        "🧾 Descrição das Características do Dataset\n\n"
        "O dataset analisado pertence ao domínio da segurança de redes IoT (Internet of Things), com foco na deteção e análise de ataques cibernéticos. "
        "Contém registos de tráfego de rede com várias métricas associadas a pacotes e conexões, tanto benignos como maliciosos.\n\n"
        "📂 Domínio: Segurança em redes IoT\n"
        f"📊 Tamanho do dataset: Aproximadamente {df.shape[0]} linhas (instâncias) e {df.shape[1]} colunas (atributos).\n"
        "📄 Tipo de ficheiro: CSV (valores separados por vírgulas)\n\n"
        "📌 Atributos Importantes:\n"
        "- proto (Categórica): Protocolo de rede utilizado (ex: TCP, UDP, ICMP).\n"
        "- flow_duration (Numérica): Duração da sessão de rede.\n"
        "- fwd_pkts_tot, bwd_pkts_tot (Numéricas): Total de pacotes enviados e recebidos.\n"
        "- fwd_data_pkts_tot, bwd_data_pkts_tot (Numéricas): Pacotes de dados úteis enviados e recebidos.\n"
        "- active.min, idle.max (Numéricas): Métricas de tempo entre pacotes e sessões ativas/inativas.\n"
        "- fwd_init_window_size, bwd_init_window_size (Numéricas): Tamanhos iniciais da janela TCP.\n"
        "- Attack_type (Categórica): Tipo de ataque identificado (ex: Normal, DDoS, MITM, MQTT_Publish).\n\n"
        "🌐 Protocolos de Rede:\n"
        "- TCP (Transmission Control Protocol): Protocolo orientado à conexão, usado para garantir entrega confiável de dados.\n"
        "- UDP (User Datagram Protocol): Protocolo rápido e sem verificação de erros, usado para vídeos em tempo real, DNS, VoIP.\n"
        "- ICMP (Internet Control Message Protocol): Utilizado para diagnóstico de rede (ex: comandos ping e traceroute). Frequentemente usado em ataques de reconhecimento.\n"
        "- MQTT (Message Queuing Telemetry Transport): Protocolo leve, usado frequentemente em IoT para comunicação publish/subscribe. Muito visado por ataques devido à sua simplicidade.\n\n"
        "🛡️ Tipos de Ataques Presentes:\n"
        "- Normal: Comunicação legítima.\n"
        "- DDoS: Envolve múltiplos dispositivos a enviarem tráfego para sobrecarregar o alvo.\n"
        "- Brute Force: Tentativas repetidas de adivinhar credenciais de login.\n"
        "- Port Scan: Verifica portas abertas para identificar serviços vulneráveis.\n"
        "- Botnet: Tráfego gerado por redes de bots controlados remotamente.\n"
        "- Web Attack: Exploração de vulnerabilidades em aplicações web (ex: SQL Injection).\n"
        "- MQTT_Publish: Ataques relacionados com o protocolo MQTT.\n"
        "- MITM (Man-in-the-Middle): Interceptação de comunicações entre dois dispositivos.\n\n"
        "📦 Pacotes de Rede:\n"
        "As métricas como fwd_pkts_tot, bwd_data_pkts_tot e flow_duration ajudam a identificar comportamentos anormais, como:\n"
        "- Envio excessivo de pacotes (comum em ataques DDoS).\n"
        "- Conexões muito curtas e repetitivas (comum em scanning).\n"
        "- Pacotes com tamanhos fora do padrão, que podem indicar tentativas de evasão de deteção.\n"
        "- A duração da sessão e o número de bytes ajudam a distinguir entre tráfego legítimo e malicioso."
    )

    # Create a new window for displaying the characteristics
    characteristics_window = tk.Toplevel(root)
    characteristics_window.title("Características do Dataset")
    characteristics_window.geometry("800x600")
    characteristics_window.resizable(True, True)

    # Add a scrollable text widget
    text_widget = tk.Text(characteristics_window, wrap=tk.WORD, font=("Arial", 12))
    text_widget.insert(tk.END, characteristics)
    text_widget.config(state=tk.DISABLED)  # Make the text read-only
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Add a scrollbar
    scrollbar = tk.Scrollbar(text_widget, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

def show_all_variables():
    """Displays all variables in the dataset with their explanations in a maximizable window."""
    variables_info = (
        "📋 Lista de Variáveis e Suas Explicações\n\n"
        "- proto (Categórica): Protocolo de rede utilizado (ex: TCP, UDP, ICMP).\n"
        "- pkt_size (Numérica): Tamanho médio dos pacotes de rede em bytes.\n"
        "- tot_pkts (Numérica): Número total de pacotes trocados numa ligação.\n"
        "- tot_bytes (Numérica): Quantidade total de dados transmitidos (em bytes).\n"
        "- flow_duration (Numérica): Duração da sessão de rede.\n"
        "- fwd_pkts_tot (Numérica): Total de pacotes enviados.\n"
        "- bwd_pkts_tot (Numérica): Total de pacotes recebidos.\n"
        "- fwd_data_pkts_tot (Numérica): Pacotes de dados úteis enviados.\n"
        "- bwd_data_pkts_tot (Numérica): Pacotes de dados úteis recebidos.\n"
        "- fwd_pkts_per_sec (Numérica): Taxa de pacotes enviados por segundo.\n"
        "- bwd_pkts_per_sec (Numérica): Taxa de pacotes recebidos por segundo.\n"
        "- payload_bytes_per_second (Numérica): Taxa de transmissão de dados úteis (bytes por segundo).\n"
        "- active.min (Numérica): Tempo mínimo de atividade entre pacotes.\n"
        "- active.avg (Numérica): Tempo médio de atividade entre pacotes.\n"
        "- active.max (Numérica): Tempo máximo de atividade entre pacotes.\n"
        "- idle.min (Numérica): Tempo mínimo de inatividade entre pacotes.\n"
        "- idle.avg (Numérica): Tempo médio de inatividade entre pacotes.\n"
        "- idle.max (Numérica): Tempo máximo de inatividade entre pacotes.\n"
        "- fwd_init_window_size (Numérica): Tamanho inicial da janela TCP para pacotes enviados.\n"
        "- bwd_init_window_size (Numérica): Tamanho inicial da janela TCP para pacotes recebidos.\n"
        "- down_up_ratio (Numérica): Razão entre pacotes recebidos e enviados.\n"
        "- flow_FIN_flag_count (Numérica): Número de pacotes com a flag FIN ativada.\n"
        "- flow_SYN_flag_count (Numérica): Número de pacotes com a flag SYN ativada.\n"
        "- flow_RST_flag_count (Numérica): Número de pacotes com a flag RST ativada.\n"
        "- flow_ACK_flag_count (Numérica): Número de pacotes com a flag ACK ativada.\n"
        "- flow_CWR_flag_count (Numérica): Número de pacotes com a flag CWR ativada.\n"
        "- flow_ECE_flag_count (Numérica): Número de pacotes com a flag ECE ativada.\n"
        "- fwd_PSH_flag_count (Numérica): Número de pacotes enviados com a flag PSH ativada.\n"
        "- bwd_PSH_flag_count (Numérica): Número de pacotes recebidos com a flag PSH ativada.\n"
        "- fwd_URG_flag_count (Numérica): Número de pacotes enviados com a flag URG ativada.\n"
        "- bwd_URG_flag_count (Numérica): Número de pacotes recebidos com a flag URG ativada.\n"
        "- fwd_iat.avg (Numérica): Tempo médio entre pacotes enviados consecutivamente.\n"
        "- fwd_iat.std (Numérica): Desvio padrão do tempo entre pacotes enviados consecutivamente.\n"
        "- bwd_iat.avg (Numérica): Tempo médio entre pacotes recebidos consecutivamente.\n"
        "- bwd_iat.std (Numérica): Desvio padrão do tempo entre pacotes recebidos consecutivamente.\n"
        "- Attack_type (Categórica): Tipo de ataque identificado (ex: Normal, DDoS, MITM, MQTT_Publish).\n"
        "- label (Categórica): Indica se o tráfego é benigno (Normal) ou malicioso (Attack).\n"
        "- src_port (Numérica): Porta de origem da conexão.\n"
        "- dst_port (Numérica): Porta de destino da conexão.\n"
        "- flags (Categórica): Flags TCP associadas à conexão, que indicam estado/controle do tráfego."
    )

    # Create a new window for displaying the variables
    variables_window = tk.Toplevel(root)
    variables_window.title("Lista de Variáveis e Explicações")
    variables_window.geometry("800x600")
    variables_window.resizable(True, True)

    # Add a scrollable text widget
    text_widget = tk.Text(variables_window, wrap=tk.WORD, font=("Arial", 12))
    text_widget.insert(tk.END, variables_info)
    text_widget.config(state=tk.DISABLED)  # Make the text read-only
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Add a scrollbar
    scrollbar = tk.Scrollbar(text_widget, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Main Tkinter window
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "RT_IOT2022.csv")
    df = load_csv(file_path)
    # Calculate down/up ratio
    df = calculate_down_up_ratio(df)
    root = tk.Tk()
    root.title("Statistical Analysis Interface")
    root.geometry("1200x800")  # Initial size
    root.configure(bg="white")  # Set background color

    # Configure grid weights for responsiveness
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Title Label
    title_label = tk.Label(
        root, 
        text="Statistical Analysis Dashboard", 
        font=("Arial", 24, "bold"), 
        bg="lightblue", 
        fg="black", 
        pady=15
    )
    title_label.pack(fill=tk.X, pady=10)

    # Create a frame for the centered layout
    main_frame = tk.Frame(root, bg="white")
    main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    # Configure grid weights for the main frame
    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_columnconfigure(1, weight=1)
    main_frame.grid_columnconfigure(2, weight=1)

    # Create a grid layout for the three sections
    general_column = tk.Frame(main_frame, bg="white", padx=10, pady=10, relief="solid", bd=1)
    general_column.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

    graphical_column = tk.Frame(main_frame, bg="white", padx=10, pady=10, relief="solid", bd=1)
    graphical_column.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

    advanced_column = tk.Frame(main_frame, bg="white", padx=10, pady=10, relief="solid", bd=1)
    advanced_column.grid(row=0, column=2, padx=20, pady=10, sticky="nsew")

    # General Analysis Section
    tk.Label(
        general_column, 
        text="General Analysis", 
        font=("Arial", 16, "bold"), 
        bg="white", 
        fg="black"
    ).pack(pady=10) 

    tk.Button(
        general_column, 
        text="Summary Statistics", 
        command=lambda: show_summary_statistics(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        general_column, 
        text="Flow Duration Distribution", 
        command=lambda: analyze_distribution(
            df, 'flow_duration', 'Flow Duration Distribution', 'Flow Duration', 'Frequency', bins=30, color='blue'
        ), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        general_column, 
        text="Dataset Characteristics", 
        command=show_dataset_characteristics, 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5)

    tk.Button(
        general_column, 
        text="List of Variables and Explanations", 
        command=show_all_variables, 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5)

    # Graphical Analyses Section
    tk.Label(
        graphical_column, 
        text="Graphical Analyses", 
        font=("Arial", 16, "bold"), 
        bg="white", 
        fg="black"
    ).pack(pady=10) 

    tk.Button(
        graphical_column, 
        text="Attack Type Distribution", 
        command=lambda: show_attack_type_distribution(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        graphical_column, 
        text="Correlation Heatmap", 
        command=lambda: show_correlation_heatmap(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        graphical_column, 
        text="Protocol Distribution", 
        command=lambda: show_protocol_distribution(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        graphical_column, 
        text="Packet Size Bar Plot", 
        command=lambda: show_packet_size_barplots(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        graphical_column, 
        text="Received Packets vs Flow Duration", 
        command=lambda: show_bwd_pkts_vs_flow_duration(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    # Add a new button to explain the importance of the graphs
    def explain_graphs():
        """Displays explanations for the importance of the graphs."""
        explanation = (
            "📊 Importance of Graphical Analyses\n\n"
            "1. **Attack Type Distribution**:\n"
            "   - Helps identify the frequency of different attack types in the dataset.\n"
            "   - Useful for understanding the prevalence of specific attacks and prioritizing mitigation strategies.\n\n"
            "2. **Correlation Heatmap**:\n"
            "   - Shows the relationships between numerical features.\n"
            "   - Helps identify highly correlated features, which can be useful for feature selection in machine learning models.\n\n"
            "3. **Protocol Distribution**:\n"
            "   - Displays the usage of different network protocols (e.g., TCP, UDP, ICMP).\n"
            "   - Useful for identifying unusual protocol usage, which may indicate malicious activity.\n\n"
            "4. **Packet Size Bar Plot**:\n"
            "   - Shows the average packet sizes for different attack types.\n"
            "   - Helps identify patterns in packet sizes that may be indicative of specific attacks.\n\n"
            "5. **Received Packets vs Flow Duration**:\n"
            "   - Visualizes the relationship between the number of received packets and the duration of network flows.\n"
            "   - Useful for identifying anomalies, such as unusually long flows with few packets, which may indicate scanning or reconnaissance activities."
        )

        # Create a new window for displaying the explanation
        explanation_window = tk.Toplevel(root)
        explanation_window.title("Importance of Graphical Analyses")
        explanation_window.geometry("800x600")
        explanation_window.resizable(True, True)

        # Add a scrollable text widget
        text_widget = tk.Text(explanation_window, wrap=tk.WORD, font=("Arial", 12))
        text_widget.insert(tk.END, explanation)
        text_widget.config(state=tk.DISABLED)  # Make the text read-only
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a scrollbar
        scrollbar = tk.Scrollbar(text_widget, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    tk.Button(
        graphical_column, 
        text="Explain Graphs and Importance", 
        command=explain_graphs, 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5)

    # Advanced Analyses Section
    tk.Label(
        advanced_column, 
        text="Advanced Analyses", 
        font=("Arial", 16, "bold"), 
        bg="white", 
        fg="black"
    ).pack(pady=10) 

    tk.Button(
        advanced_column, 
        text="Analyze Flags", 
        command=lambda: analyze_flags(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        advanced_column, 
        text="Analyze IAT", 
        command=lambda: analyze_iat(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        advanced_column, 
        text="Analyze Activity and Idle Times", 
        command=lambda: analyze_activity_idle(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        advanced_column, 
        text="Analyze Down/Up Ratio", 
        command=lambda: analyze_down_up_ratio(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 
        
    # Run the Tkinter main loop
    root.mainloop()
