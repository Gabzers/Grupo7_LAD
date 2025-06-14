import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Import os for file existence check
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from joblib import load, dump
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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

# Additional graphical Analysis
def additional_graphical_analysis(df):
    """Performs additional graphical Analysis based on the dataset."""
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
        "mode": "Mode (most frequent value)"
    }

    # Create a frame for explanations
    explanation_frame = tk.Frame(stats_window)
    explanation_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    tk.Label(explanation_frame, text="Explanations of Statistics:", font=("Arial", 12, "bold")).pack(anchor="w")
    for metric, explanation in explanations.items():
        tk.Label(explanation_frame, text=f"{metric}: {explanation}", font=("Arial", 10)).pack(anchor="w")

    # Add the number of variables to the analysis
    num_variables = len(df.columns)
    tk.Label(
        explanation_frame,
        text=f"Number of Variables: {num_variables}",
        font=("Arial", 12, "bold"),
        fg="blue"
    ).pack(anchor="w", pady=(10, 0))

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

# Functions for individual graphical Analysis
def show_attack_type_distribution(df):
    """Displays the distribution of attack types."""
    plt.figure(figsize=(12, 8))  # Increased figure size for better clarity
    sns.countplot(
        data=df, 
        y="Attack_type", 
        order=df["Attack_type"].value_counts().index, 
        palette="coolwarm"
    )
    
    # Enhanced titles and labels
    plt.title("Distribution of Attack Types", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Occurrences", fontsize=12)
    plt.ylabel("Attack Type", fontsize=12)
    
    # Layout adjustment to prevent element overlap
    plt.tight_layout()

    plt.show()

def show_correlation_heatmap(df):
    """Displays the correlation matrix for selected features."""
    plt.figure(figsize=(12, 10))  # Increased size for better visualization

    # Create a dictionary to map the original variable names to more understandable ones
    feature_names = {
        "flow_duration": "Flow Duration",
        "fwd_pkts_tot": "Total Forward Packets",
        "bwd_pkts_tot": "Total Backward Packets",
        "flow_pkts_per_sec": "Packets per Second",
        "payload_bytes_per_second": "Payload Bytes per Second",
        "idle.avg": "Average Idle Time",
        "active.avg": "Average Active Time"
    }

    selected_features = list(feature_names.keys())
    
    # Calculate the correlation matrix
    corr_matrix = df[selected_features].corr()
    
    # Create the heatmap with enhanced settings
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=0.5, 
        cbar_kws={"label": "Correlation Coefficient"}  # Add label to color bar
    )
    
    # Rotate X-axis labels with translated names
    plt.xticks(ticks=range(len(selected_features)), labels=[feature_names[feature] for feature in selected_features], rotation=45)
    plt.yticks(ticks=range(len(selected_features)), labels=[feature_names[feature] for feature in selected_features])

    # Enhanced titles and layout
    plt.title("Correlation Matrix for Selected Features", fontsize=14, fontweight="bold")
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()  # Adjust layout to prevent overlap
    
    plt.show()



import matplotlib.pyplot as plt

def show_protocol_distribution(df):
    """Displays the distribution of network protocols."""
    plt.figure(figsize=(10, 8))  # Increased figure size for better clarity
    
    # Create a pie chart with adjustments for better readability
    df["proto"].value_counts().head(10).plot.pie(
        autopct="%1.1f%%", 
        cmap="tab10", 
        startangle=90,
        pctdistance=0.85  # Adjust the position of the percentage for better visibility
    )
    
    plt.title("Distribution of Network Protocols", fontsize=14, fontweight="bold")
    plt.ylabel("")  # Remove the default y-axis label
    
    # Add a legend outside the chart
    plt.legend(
        title="Protocols", 
        loc="center left", 
        bbox_to_anchor=(1, 0.5),  # Position the legend outside the chart
        fontsize=10
    )
    
    plt.tight_layout()  # Adjust the layout to avoid overlap
    plt.show()


def show_packet_size_barplots(df):
    """Displays enhanced bar plots for packet size variables by attack type."""
    variables = ['fwd_pkts_payload.tot', 'bwd_pkts_payload.tot', 'fwd_pkts_tot', 'bwd_pkts_tot']
    
    for var in variables:
        plt.figure(figsize=(14, 8))  # Increased figure size for better visibility
        avg_packet_size = df.groupby("Attack_type")[var].mean().sort_values(ascending=False)
        
        avg_packet_size.plot(
            kind="bar", 
            color="skyblue", 
            edgecolor="black"  # Added edges for better element definition
        )
        
        # Enhanced titles and labels
        plt.title(f"Average of {var.replace('_', ' ').title()} by Attack Type", fontsize=16, fontweight="bold")
        plt.xlabel("Attack Type", fontsize=14)
        plt.ylabel(f"Average of {var.replace('_', ' ').title()}", fontsize=14)
        
        # Better label readability
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add description to the graph
        plt.text(
            0.95, 0.01, 
            "Analysis of data grouped by attack type", 
            fontsize=10, 
            transform=plt.gcf().transFigure, 
            ha="right"
        )
        
        plt.tight_layout()  # Adjust layout to avoid overlaps
        plt.show()
        plt.close()  # Close the plot to free memory


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

def show_hexbin_only(df):
    plt.figure(figsize=(12, 8))
    plt.hexbin(
        df["bwd_pkts_tot"],
        df["flow_duration"],
        gridsize=50,
        cmap="coolwarm",
        bins='log'
    )
    plt.colorbar(label='Flow Count (log)')
    plt.title("Density Between Received Packets and Flow Duration", fontsize=16, fontweight="bold")
    plt.xlabel("Received Packets (Total)", fontsize=14)
    plt.ylabel("Flow Duration", fontsize=14)
    plt.tight_layout()
    plt.show()

def show_attack_distribution_pie(df):
    """Displays the distribution of attack types as a pie chart."""
    attack_counts = df["Attack_type"].value_counts()
    explode = [0.1 if i == 0 else 0.05 for i in range(len(attack_counts.head(10)))]  # Slightly separate all slices
    plt.figure(figsize=(14, 10))  # Increased figure size for better clarity

    # Create a pie chart without percentages and names
    attack_counts.head(10).plot(
        kind="pie",
        startangle=90,
        cmap="tab20",
        explode=explode,
        labels=None,  # Remove labels from the pie chart
    )
    
    plt.title("Attack Type Distribution (Pie Chart)", fontsize=16, fontweight="bold")
    plt.ylabel("")  # Remove the default y-axis label
    
    # Add a legend outside the chart with percentages
    legend_labels = [
        f"{label} ({attack_counts[label] / attack_counts.sum() * 100:.1f}%)"
        for label in attack_counts.head(10).index
    ]
    plt.legend(
        labels=legend_labels,
        title="Attack Types",
        loc="center left",
        bbox_to_anchor=(1, 0.5),  # Position the legend outside the chart
        fontsize=10
    )
    
    plt.tight_layout()  # Adjust the layout to avoid overlap
    plt.show()

def analyze_each_graph():
    """Provides an analysis of each graph and its insights."""
    analysis = (
        "📊 Analysis of Each Graph and Its Insights\n\n"
        "1. **Attack Type Distribution**:\n"
        "   - Insight: This graph shows the frequency of each attack type in the dataset.\n"
        "   - Analysis: A high frequency of certain attack types (e.g., DDoS) may indicate that the dataset is imbalanced. "
        "This imbalance can affect the performance of machine learning models and may require techniques like oversampling or undersampling.\n\n"
        "   - Observations:\n"
        "     - **DOS_SYN_Hping** stands out massively compared to the others, with over 90,000 occurrences.\n"
        "       - This attack is a type of Denial-of-Service (DoS) that exploits the TCP protocol using SYN packets (likely generated by the Hping tool).\n"
        "       - The fact that this type represents the absolute majority suggests:\n"
        "         - A specific attack campaign in the dataset.\n"
        "         - Possible data imbalance in the database (high asymmetry).\n"
        "         - High relevance of this attack vector for the analyzed systems.\n"
        "     - **Intermediate Group**:\n"
        "       - Includes Thing_Speak, ARP_poisoning, and MQTT_Publish, with visibly lower values (~5,000 to ~10,000) but still significant.\n"
        "       - ARP_poisoning refers to ARP spoofing attacks – used to intercept or redirect local network traffic.\n"
        "       - MQTT_Publish and Thing_Speak suggest attacks or malicious communications in IoT (Internet of Things) systems.\n"
        "     - **Scanning Attacks**:\n"
        "       - NMAP_UDP_SCAN, NMAP_TCP_scan, NMAP_FIN_SCAN, XMAS_TREE_SCAN, IP_OS_DETECTION represent reconnaissance and network mapping activities.\n"
        "       - These are common in the initial attack phase, where the attacker tries to identify open ports and services.\n"
        "     - **Low-Frequency Attacks**:\n"
        "       - DDOS_Slowloris, Wipro_bulb, Bot_Brute_Force_SSH, NMAP_FIN_SCAN have very small occurrences.\n"
        "       - Although rare, these attacks can be highly dangerous or indicative of specialized exploitation.\n\n"
        "   - Conclusion:\n"
        "     - The graph shows an absolute predominance of DoS attacks with SYN flooding via Hping, indicating a possible focus of this dataset or a real simulation scenario.\n"
        "     - The presence of IoT system attacks and scanning suggests diversity in attack vectors.\n"
        "     - It is recommended to perform a temporal analysis (e.g., how attacks evolve over time) and correlate with other data, such as IPs, ports, protocols, or locations.\n\n"
        "2. **Correlation Heatmap**:\n"
        "   - Insight: Displays the relationships between numerical features.\n"
        "   - Analysis: Strong correlations (e.g., >0.8 or <-0.8) between features can indicate redundancy. "
        "Highly correlated features can be removed to reduce dimensionality without losing much information.\n\n"
        "   - Observations:\n"
        "     - **Very High Correlation (0.98)**:\n"
        "       - Between flow_pkts_per_sec and payload_bytes_per_second. This indicates that both measure very similar information about the transmission rate, suggesting redundancy.\n"
        "     - **Moderate Correlation (0.74)**:\n"
        "       - Between flow_duration and fwd_pkts_tot, indicating that longer flows tend to have more packets sent.\n"
        "     - **Weak or Insignificant Correlations**:\n"
        "       - For example, between flow_duration and flow_pkts_per_sec (-0.03), showing that longer flows do not necessarily have a higher packet sending rate.\n"
        "     - Other variables, such as active.avg and idle.avg, have low correlation with others, potentially capturing independent aspects of traffic behavior.\n\n"
        "   - Conclusion:\n"
        "     - The analysis highlights the need to address high correlation (redundancy) between certain features to avoid issues such as overfitting in predictive models or subsequent statistical Analysis.\n\n"
        "3. **Protocol Distribution**:\n"
        "   - Insight: Shows the proportion of different network protocols (e.g., TCP, UDP, ICMP).\n"
        "   - Analysis: Unusual protocol usage (e.g., a high percentage of ICMP traffic) may indicate reconnaissance or scanning activities.\n\n"
        "   - Observations:\n"
        "     - **TCP (89.7%)**: Dominates the network, reflecting intensive use of services requiring reliable transmission, such as web browsing and data transfer.\n"
        "     - **UDP (10.3%)**: Represents a significant portion, used by applications prioritizing speed and low latency, such as streaming and VoIP.\n"
        "     - **ICMP (0.0%)**: Appears negligibly, typically used for network diagnostics (e.g., pings).\n"
        "       - In many networks, TCP and UDP can cover almost all application traffic.\n"
        "       - Layer 3 protocols, such as ICMP, may appear less frequently compared to transport protocols (TCP/UDP) that carry payloads in higher volume.\n\n"
        "   - Conclusion:\n"
        "     - This pie chart shows that your network/dataset is predominantly TCP-based, with a notable UDP component. ICMP is residual.\n"
        "     - If more detail is needed, we could:\n"
        "       - Examine temporal variables (TCP/UDP flow over time).\n"
        "       - Analyze subprotocols or services (e.g., DNS, HTTP, HTTPS).\n"
        "       - Investigate anomalies, uncommon ports, or suspicious traffic.\n\n"
        "4. **Packet Size Bar Plot**:\n"
        "- **Insight**: This bar plot displays the average forward packet payload size (`fwd_pkts_payload.tot`) for different attack types.\n"
        "- **Analysis**:\n"
        "  - The graph highlights the variation in average packet payload sizes across attack types, revealing distinct patterns that can aid in attack identification.\n"
        "- **Observations**:\n"
        "  - **Wipro_bulb**: This attack type exhibits the highest average payload size, suggesting large data transfers, possibly indicative of data exfiltration or bulk communication.\n"
        "  - **ARP_poisoning and Metasploit_Brute_Force_SSH**: These attacks have moderate payload sizes, reflecting their nature as targeted attacks with controlled data exchange.\n"
        "  - **Thing_Speak, DDOS_Slowloris, and DOS_SYN_Hping**: These attacks show relatively small average payload sizes, consistent with their nature as volumetric or low-bandwidth attacks.\n"
        "  - **NMAP Scans**: These scanning activities have minimal payload sizes, as they primarily involve reconnaissance with minimal data exchange.\n"
        "- **Conclusion**:\n"
        "  - The graph provides valuable insights into the nature of different attack types based on their average packet payload sizes.\n"
        "  - The distinct patterns observed can aid in the development of targeted detection and mitigation strategies for specific attack types.\n\n"
        "5. **Hexbin: Packets vs Duration**:\n"
        "   - Insight: This scatterplot visualizes the relationship between the total number of received packets (`bwd_pkts_tot`) and the duration of network flows (`flow_duration`).\n"
        "   - Analysis: The graph highlights patterns in traffic behavior for different attack types. Clusters of points may indicate specific attack characteristics, such as:\n"
        "     - **Short flows with few packets**: Likely reconnaissance or scanning activities.\n"
        "     - **Long flows with many packets**: Indicative of sustained attacks, such as DDoS or data exfiltration.\n"
        "     - **Outliers**: Points with extremely high flow duration or packet counts may represent anomalies or specific attack scenarios.\n\n"
        "   - Observations:\n"
        "     - **MQTT_Publish and Thing_Speak**: These attack types dominate the upper-left region, with long flow durations but relatively few packets.\n"
        "     - **NMAP Scans**: Concentrated in the lower-left region, indicating short flows with minimal packet exchange, typical of scanning activities.\n"
        "     - **DOS_SYN_Hping**: Scattered across the plot, showing variability in flow duration and packet counts, consistent with its nature as a volumetric attack.\n"
        "     - **Outliers**: A few points with extremely high flow durations suggest prolonged sessions, possibly due to persistent attacks or misconfigurations.\n\n"
        "   - Conclusion:\n"
        "     - The graph reveals distinct traffic patterns for different attack types, aiding in their identification and classification.\n"
        "     - The presence of outliers and clusters suggests the need for further investigation into specific attack behaviors.\n"
        "     - This visualization is valuable for understanding traffic dynamics and designing targeted mitigation strategies.\n"
    )

    # Create a new window for displaying the analysis
    analysis_window = tk.Toplevel(root)
    analysis_window.title("Analysis of Each Graph")
    analysis_window.geometry("800x600")
    analysis_window.resizable(True, True)

    # Add a scrollable text widget
    text_widget = tk.Text(analysis_window, wrap=tk.WORD, font=("Arial", 12))
    text_widget.insert(tk.END, analysis)
    text_widget.config(state=tk.DISABLED)  # Make the text read-only
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Add a scrollbar
    scrollbar = tk.Scrollbar(text_widget, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Add the button to the Graphical Analysis section
   

# Add down_up_ratio calculation
def calculate_down_up_ratio(df):
    """Calculates the down/up ratio for the dataset."""
    df['down_up_ratio'] = df['bwd_pkts_tot'] / (df['fwd_pkts_tot'] + 1e-9)  # Avoid division by zero
    return df


def show_dataset_characteristics():
    """Displays the characteristics of the dataset in a maximizable window."""
    characteristics = (
        "🧾 Description of Dataset Characteristics\n\n"
        "The analyzed dataset belongs to the domain of IoT (Internet of Things) network security, focusing on the detection and analysis of cyberattacks. "
        "It contains records of network traffic with various metrics associated with packets and connections, both benign and malicious.\n\n"
        "📂 Domain: IoT Network Security\n"
        f"📊 Dataset size: Approximately {df.shape[0]} rows (instances) and {df.shape[1]} columns (attributes).\n"
        "📄 File type: CSV (comma-separated values)\n\n"
        "📌 Important Attributes:\n"
        "- proto (Categorical): Network protocol used (e.g., TCP, UDP, ICMP).\n"
        "- flow_duration (Numeric): Duration of the network session.\n"
        "- fwd_pkts_tot, bwd_pkts_tot (Numeric): Total packets sent and received.\n"
        "- fwd_data_pkts_tot, bwd_data_pkts_tot (Numeric): Useful data packets sent and received.\n"
        "- active.min, idle.max (Numeric): Time metrics between packets and active/inactive sessions.\n"
        "- fwd_init_window_size, bwd_init_window_size (Numeric): Initial TCP window sizes.\n"
        "- Attack_type (Categorical): Identified attack type (e.g., Normal, DDoS, MITM, MQTT_Publish).\n\n"
        "🌐 Network Protocols:\n"
        "- TCP (Transmission Control Protocol): Connection-oriented protocol used to ensure reliable data delivery.\n"
        "- UDP (User Datagram Protocol): Fast and error-check-free protocol used for real-time videos, DNS, VoIP.\n"
        "- ICMP (Internet Control Message Protocol): Used for network diagnostics (e.g., ping and traceroute commands). Often used in reconnaissance attacks.\n"
        "- MQTT (Message Queuing Telemetry Transport): Lightweight protocol, often used in IoT for publish/subscribe communication. Highly targeted by attacks due to its simplicity.\n\n"
        "🛡️ Types of Attacks Present:\n"
        "- Normal: Legitimate communication.\n"
        "- DDoS: Involves multiple devices sending traffic to overwhelm the target.\n"
        "- Brute Force: Repeated attempts to guess login credentials.\n"
        "- Port Scan: Scans open ports to identify vulnerable services.\n"
        "- Botnet: Traffic generated by remotely controlled bot networks.\n"
        "- Web Attack: Exploitation of vulnerabilities in web applications (e.g., SQL Injection).\n"
        "- MQTT_Publish: Attacks related to the MQTT protocol.\n"
        "- MITM (Man-in-the-Middle): Interception of communications between two devices.\n\n"
        "📦 Network Packets:\n"
        "Metrics such as fwd_pkts_tot, bwd_data_pkts_tot, and flow_duration help identify abnormal behaviors, such as:\n"
        "- Excessive packet sending (common in DDoS attacks).\n"
        "- Very short and repetitive connections (common in scanning).\n"
        "- Packets with out-of-standard sizes, which may indicate evasion attempts.\n"
        "- Session duration and the number of bytes help distinguish between legitimate and malicious traffic."
    )

    # Create a new window for displaying the characteristics
    characteristics_window = tk.Toplevel(root)
    characteristics_window.title("Dataset Characteristics")
    characteristics_window.geometry("800x600")
    characteristics_window.resizable(True, True)

    # Add a scrollable text widget
    text_widget = tk.Text(characteristics_window, wrap=tk.WORD, font=("Arial", 12))
    text_widget.insert(tk.END, characteristics)
    text_widget.config(state=tk.DISABLED)  # Make the text read-only
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

def show_all_variables():
    """Displays all variables in the dataset with their explanations in a maximizable window."""
    variables_info = (
        "📋 List of Variables and Their Explanations\n\n"
        "- proto (Categorical): Network protocol used (e.g., TCP, UDP, ICMP).\n"
        "- pkt_size (Numeric): Average size of network packets in bytes.\n"
        "- tot_pkts (Numeric): Total number of packets exchanged in a connection.\n"
        "- tot_bytes (Numeric): Total amount of data transmitted (in bytes).\n"
        "- flow_duration (Numeric): Duration of the network session.\n"
        "- fwd_pkts_tot (Numeric): Total packets sent.\n"
        "- bwd_pkts_tot (Numeric): Total packets received.\n"
        "- fwd_data_pkts_tot (Numeric): Useful data packets sent.\n"
        "- bwd_data_pkts_tot (Numeric): Useful data packets received.\n"
        "- fwd_pkts_per_sec (Numeric): Rate of packets sent per second.\n"
        "- bwd_pkts_per_sec (Numeric): Rate of packets received per second.\n"
        "- payload_bytes_per_second (Numeric): Rate of useful data transmission (bytes per second).\n"
        "- active.min (Numeric): Minimum active time between packets.\n"
        "- active.avg (Numeric): Average active time between packets.\n"
        "- active.max (Numeric): Maximum active time between packets.\n"
        "- idle.min (Numeric): Minimum idle time between packets.\n"
        "- idle.avg (Numeric): Average idle time between packets.\n"
        "- idle.max (Numeric): Maximum idle time between packets.\n"
        "- fwd_init_window_size (Numeric): Initial TCP window size for sent packets.\n"
        "- bwd_init_window_size (Numeric): Initial TCP window size for received packets.\n"
        "- down_up_ratio (Numeric): Ratio between received and sent packets.\n"
        "- flow_FIN_flag_count (Numeric): Number of packets with the FIN flag activated.\n"
        "- flow_SYN_flag_count (Numeric): Number of packets with the SYN flag activated.\n"
        "- flow_RST_flag_count (Numeric): Number of packets with the RST flag activated.\n"
        "- flow_ACK_flag_count (Numeric): Number of packets with the ACK flag activated.\n"
        "- flow_CWR_flag_count (Numeric): Number of packets with the CWR flag activated.\n"
        "- flow_ECE_flag_count (Numeric): Number of packets with the ECE flag activated.\n"
        "- fwd_PSH_flag_count (Numeric): Number of sent packets with the PSH flag activated.\n"
        "- bwd_PSH_flag_count (Numeric): Number of received packets with the PSH flag activated.\n"
        "- fwd_URG_flag_count (Numeric): Number of sent packets with the URG flag activated.\n"
        "- bwd_URG_flag_count (Numeric): Number of received packets with the URG flag activated.\n"
        "- fwd_iat.avg (Numeric): Average time between consecutively sent packets.\n"
        "- fwd_iat.std (Numeric): Standard deviation of the time between consecutively sent packets.\n"
        "- bwd_iat.avg (Numeric): Average time between consecutively received packets.\n"
        "- bwd_iat.std (Numeric): Standard deviation of the time between consecutively received packets.\n"
        "- id.orig_p (Numeric): Source port of the connection.\n"
        "- id.resp_p (Numeric): Destination port of the connection.\n"
        "- proto (Categorical): Protocol used in the connection (e.g., TCP, UDP).\n"
        "- service (Categorical): Service associated with the connection (e.g., HTTP, FTP).\n"
        "- flow_duration (Numeric): Duration of the network flow in microseconds.\n"
        "- fwd_pkts_tot (Numeric): Total number of packets sent in the forward direction.\n"
        "- bwd_pkts_tot (Numeric): Total number of packets received in the backward direction.\n"
        "- fwd_data_pkts_tot (Numeric): Total number of data packets sent in the forward direction.\n"
        "- bwd_data_pkts_tot (Numeric): Total number of data packets received in the backward direction.\n"
        "- fwd_pkts_per_sec (Numeric): Rate of packets sent per second in the forward direction.\n"
        "- bwd_pkts_per_sec (Numeric): Rate of packets received per second in the backward direction.\n"
        "- flow_pkts_per_sec (Numeric): Total rate of packets per second in the flow.\n"
        "- down_up_ratio (Numeric): Ratio of received to sent packets.\n"
        "- fwd_header_size_tot (Numeric): Total size of headers in the forward direction.\n"
        "- fwd_header_size_min (Numeric): Minimum size of headers in the forward direction.\n"
        "- fwd_header_size_max (Numeric): Maximum size of headers in the forward direction.\n"
        "- bwd_header_size_tot (Numeric): Total size of headers in the backward direction.\n"
        "- bwd_header_size_min (Numeric): Minimum size of headers in the backward direction.\n"
        "- bwd_header_size_max (Numeric): Maximum size of headers in the backward direction.\n"
        "- flow_FIN_flag_count (Numeric): Number of packets with the FIN flag activated.\n"
        "- flow_SYN_flag_count (Numeric): Number of packets with the SYN flag activated.\n"
        "- flow_RST_flag_count (Numeric): Number of packets with the RST flag activated.\n"
        "- fwd_PSH_flag_count (Numeric): Number of sent packets with the PSH flag activated.\n"
        "- bwd_PSH_flag_count (Numeric): Number of received packets with the PSH flag activated.\n"
        "- flow_ACK_flag_count (Numeric): Number of packets with the ACK flag activated.\n"
        "- fwd_URG_flag_count (Numeric): Number of sent packets with the URG flag activated.\n"
        "- bwd_URG_flag_count (Numeric): Number of received packets with the URG flag activated.\n"
        "- flow_CWR_flag_count (Numeric): Number of packets with the CWR flag activated.\n"
        "- flow_ECE_flag_count (Numeric): Number of packets with the ECE flag activated.\n"
        "- fwd_pkts_payload.min (Numeric): Minimum payload size of sent packets.\n"
        "- fwd_pkts_payload.max (Numeric): Maximum payload size of sent packets.\n"
        "- fwd_pkts_payload.tot (Numeric): Total payload size of sent packets.\n"
        "- fwd_pkts_payload.avg (Numeric): Average payload size of sent packets.\n"
        "- fwd_pkts_payload.std (Numeric): Standard deviation of payload size of sent packets.\n"
        "- bwd_pkts_payload.min (Numeric): Minimum payload size of received packets.\n"
        "- bwd_pkts_payload.max (Numeric): Maximum payload size of received packets.\n"
        "- bwd_pkts_payload.tot (Numeric): Total payload size of received packets.\n"
        "- bwd_pkts_payload.avg (Numeric): Average payload size of received packets.\n"
        "- bwd_pkts_payload.std (Numeric): Standard deviation of payload size of received packets.\n"
        "- flow_pkts_payload.min (Numeric): Minimum payload size of packets in the flow.\n"
        "- flow_pkts_payload.max (Numeric): Maximum payload size of packets in the flow.\n"
        "- flow_pkts_payload.tot (Numeric): Total payload size of packets in the flow.\n"
        "- flow_pkts_payload.avg (Numeric): Average payload size of packets in the flow.\n"
        "- flow_pkts_payload.std (Numeric): Standard deviation of payload size of packets in the flow.\n"
        "- fwd_iat.min (Numeric): Minimum inter-arrival time between sent packets.\n"
        "- fwd_iat.max (Numeric): Maximum inter-arrival time between sent packets.\n"
        "- fwd_iat.tot (Numeric): Total inter-arrival time between sent packets.\n"
        "- fwd_iat.avg (Numeric): Average inter-arrival time between sent packets.\n"
        "- fwd_iat.std (Numeric): Standard deviation of inter-arrival time between sent packets.\n"
        "- bwd_iat.min (Numeric): Minimum inter-arrival time between received packets.\n"
        "- bwd_iat.max (Numeric): Maximum inter-arrival time between received packets.\n"
        "- bwd_iat.tot (Numeric): Total inter-arrival time between received packets.\n"
        "- bwd_iat.avg (Numeric): Average inter-arrival time between received packets.\n"
        "- bwd_iat.std (Numeric): Standard deviation of inter-arrival time between received packets.\n"
        "- flow_iat.min (Numeric): Minimum inter-arrival time between packets in the flow.\n"
        "- flow_iat.max (Numeric): Maximum inter-arrival time between packets in the flow.\n"
        "- flow_iat.tot (Numeric): Total inter-arrival time between packets in the flow.\n"
        "- flow_iat.avg (Numeric): Average inter-arrival time between packets in the flow.\n"
        "- flow_iat.std (Numeric): Standard deviation of inter-arrival time between packets in the flow.\n"
        "- payload_bytes_per_second (Numeric): Rate of payload bytes per second in the flow.\n"
        "- fwd_subflow_pkts (Numeric): Number of subflow packets sent in the forward direction.\n"
        "- bwd_subflow_pkts (Numeric): Number of subflow packets received in the backward direction.\n"
        "- fwd_subflow_bytes (Numeric): Number of subflow bytes sent in the forward direction.\n"
        "- bwd_subflow_bytes (Numeric): Number of subflow bytes received in the backward direction.\n"
        "- fwd_bulk_bytes (Numeric): Number of bulk bytes sent in the forward direction.\n"
        "- bwd_bulk_bytes (Numeric): Number of bulk bytes received in the backward direction.\n"
        "- fwd_bulk_packets (Numeric): Number of bulk packets sent in the forward direction.\n"
        "- bwd_bulk_packets (Numeric): Number of bulk packets received in the backward direction.\n"
        "- fwd_bulk_rate (Numeric): Rate of bulk bytes sent in the forward direction.\n"
        "- bwd_bulk_rate (Numeric): Rate of bulk bytes received in the backward direction.\n"
        "- active.min (Numeric): Minimum active time of the flow.\n"
        "- active.max (Numeric): Maximum active time of the flow.\n"
        "- active.tot (Numeric): Total active time of the flow.\n"
        "- active.avg (Numeric): Average active time of the flow.\n"
        "- active.std (Numeric): Standard deviation of active time of the flow.\n"
        "- idle.min (Numeric): Minimum idle time of the flow.\n"
        "- idle.max (Numeric): Maximum idle time of the flow.\n"
        "- idle.tot (Numeric): Total idle time of the flow.\n"
        "- idle.avg (Numeric): Average idle time of the flow.\n"
        "- idle.std (Numeric): Standard deviation of idle time of the flow.\n"
        "- fwd_init_window_size (Numeric): Initial TCP window size for sent packets.\n"
        "- bwd_init_window_size (Numeric): Initial TCP window size for received packets.\n"
        "- fwd_last_window_size (Numeric): Last TCP window size for sent packets.\n"
        "- Attack_type (Categorical): Identified attack type (e.g., Normal, DDoS, MITM, MQTT_Publish).\n"
    )

    # Create a new window for displaying the variables
    variables_window = tk.Toplevel(root)
    variables_window.title("List of Variables and Explanations")
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

def predict_attack_types():
    try:
        # Carregar modelo treinado
        model = load("random_forest_model.joblib")
        # Carregar dados
        df_pred = load_csv(os.path.join(os.path.dirname(__file__), "RT_IOT2022.csv"))
        # Remover colunas desnecessárias
        if "Unnamed: 0" in df_pred.columns:
            df_pred.drop(columns=["Unnamed: 0"], inplace=True)
        # Codificar colunas categóricas (exceto target)
        categorical_cols = df_pred.select_dtypes(include=["object"]).columns.tolist()
        if "Attack_type" in categorical_cols:
            categorical_cols.remove("Attack_type")
        df_pred = pd.get_dummies(df_pred, columns=categorical_cols)
        # Codificar variável alvo (para comparar, se existir)
        le = LabelEncoder()
        if "Attack_type" in df_pred.columns:
            df_pred["Attack_type"] = le.fit_transform(df_pred["Attack_type"])
            X_pred = df_pred.drop("Attack_type", axis=1)
        else:
            X_pred = df_pred
        # Normalização (ajustar para usar scaler do treino se disponível)
        scaler = StandardScaler()
        X_pred_scaled = scaler.fit_transform(X_pred)
        # Predição
        predicoes = model.predict(X_pred_scaled)
        # Adicionar resultados ao DataFrame
        df_pred["Predicted_Attack"] = le.inverse_transform(predicoes)
        # Salvar resultados
        df_pred.to_csv("resultados_com_predicoes.csv", index=False)
        # Mostrar resumo em janela Tkinter
        summary = df_pred["Predicted_Attack"].value_counts().to_string()
        result_window = tk.Toplevel(root)
        result_window.title("Prediction Summary")
        tk.Label(result_window, text="Prediction Summary (Predicted_Attack)", font=("Arial", 14, "bold")).pack(pady=10)
        text_widget = tk.Text(result_window, wrap=tk.WORD, font=("Arial", 12), height=20, width=60)
        text_widget.insert(tk.END, summary)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(padx=10, pady=10)
        tk.Label(result_window, text="Results saved in 'resultados_com_predicoes.csv'", font=("Arial", 10, "italic"), fg="green").pack(pady=5)
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

# Main Tkinter window
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "RT_IOT2022.csv")
    df = load_csv(file_path)

    # Calculate down/up ratio
    df = calculate_down_up_ratio(df)

    # Initialize ttkbootstrap
    from ttkbootstrap import Window, Style  # Import Window and Style from ttkbootstrap
    root = Window(themename="flatly")  # Use a clean and modern light theme
    root.title("📊 Advanced IoT Security Analytics")
    root.geometry("1400x900")  # Larger size for better layout

    # Title Label
    title_label = ttk.Label(
        root,
        text="📊 Advanced IoT Security Analytics",
        font=("Helvetica", 36, "bold italic"),  # Updated font size and style
        anchor="center",
        bootstyle="inverse-primary"  # Dark blue background
    )
    title_label.pack(fill="x", pady=20)

    # Create a frame for the centered layout
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(expand=True, fill="both")

    # Configure grid weights for the main frame
    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)  # Add weight to the left spacer
    main_frame.grid_columnconfigure(1, weight=2)  # Add weight to the general analysis column
    main_frame.grid_columnconfigure(2, weight=2)  # Add weight to the graphical analysis column
    main_frame.grid_columnconfigure(3, weight=1)  # Add weight to the right spacer

    style = Style()

    # General Analysis Section
    general_column = ttk.Labelframe(main_frame, padding=15, bootstyle="primary")
    general_column.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")  # Place in column 1
    ttk.Label(
        general_column,
        text="General Analysis",  # Add title inside the rectangle
        font=("Helvetica", 18, "bold"),  # Slightly larger font
        anchor="center",
        foreground=style.colors.get("primary"),  # Dark blue background
    ).pack(pady=(5, 15))  # Add padding to separate from buttons

    # Graphical Analysis Section
    graphical_column = ttk.Labelframe(main_frame, padding=15, bootstyle="primary")
    graphical_column.grid(row=0, column=2, padx=20, pady=10, sticky="nsew")  # Place in column 2
    ttk.Label(
        graphical_column,
        text="Graphical Analysis",  # Add title inside the rectangle
        font=("Helvetica", 18, "bold"),  # Slightly larger font
        anchor="center",
        foreground=style.colors.get("primary"),  # Dark blue background
    ).pack(pady=(5, 15))  # Add padding to separate from buttons

    # General Analysis Section
    ttk.Button(
        general_column,
        text="Summary Statistics",
        command=lambda: show_summary_statistics(df),
        bootstyle="outline-primary",  # Unified color with title
    ).pack(pady=10, fill="x")

    ttk.Button(
        general_column,
        text="Dataset Characteristics",
        command=show_dataset_characteristics,
        bootstyle="outline-primary",  # Unified color with title
    ).pack(pady=10, fill="x")

    ttk.Button(
        general_column,
        text="List of Variables and Explanations",
        command=show_all_variables,
        bootstyle="outline-primary",  # Unified color with title
    ).pack(pady=10, fill="x")

    ttk.Button(
        general_column,
        text="Predict Attack Type",
        command=predict_attack_types,
        bootstyle="outline-primary",
    ).pack(pady=10, fill="x")

    # Graphical Analysis Section
    ttk.Button(
        graphical_column,
        text="Attack Type Distribution",
        command=lambda: show_attack_type_distribution(df),
        bootstyle="outline-primary",  # Unified color with title
    ).pack(pady=10, fill="x")

    ttk.Button(
        graphical_column,
        text="Attack Distribution (Pie Chart)",
        command=lambda: show_attack_distribution_pie(df),
        bootstyle="outline-primary",  # Unified color with title
    ).pack(pady=10, fill="x")

    ttk.Button(
        graphical_column,
        text="Correlation Heatmap",
        command=lambda: show_correlation_heatmap(df),
        bootstyle="outline-primary",  # Unified color with title
    ).pack(pady=10, fill="x")

    ttk.Button(
        graphical_column,
        text="Protocol Distribution",
        command=lambda: show_protocol_distribution(df),
        bootstyle="outline-primary",  # Unified color with title
    ).pack(pady=10, fill="x")

    ttk.Button(
        graphical_column,
        text="Packet Size Bar Plot",
        command=lambda: show_packet_size_barplots(df),
        bootstyle="outline-primary",  # Unified color with title
    ).pack(pady=10, fill="x")

    ttk.Button(
        graphical_column,
        text="Hexbin: Packets vs Duration",
        command=lambda: show_hexbin_only(df),
        bootstyle="outline-primary", # Unified color with title
    ).pack(pady=10, fill="x")

    ttk.Button(
        graphical_column,
        text="Analyze Each Graph",
        command=lambda: analyze_each_graph(),
        bootstyle="outline-primary",  # Unified color with title
    ).pack(pady=10, fill="x")

    # Run the Tkinter main loop
    root.mainloop()

