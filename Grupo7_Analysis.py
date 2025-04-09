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
    df["proto"].value_counts().head(10).plot(kind="pie", autopct="%1.1f%%", cmap="tab10", startangle=90)
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

def show_idle_avg_boxplot(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="proto", y="idle.avg", palette="Set2")
    plt.ylim(0, df["idle.avg"].quantile(0.99))
    plt.title("Distribution of Idle Time by Protocol")
    plt.xlabel("Protocol")
    plt.ylabel("Idle (Average)")
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
        text="Distribution of Flow Duration", 
        command=lambda: analyze_distribution(
            df, 'flow_duration', 'Flow Duration Distribution', 'Flow Duration', 'Frequency', bins=30, color='blue'
        ), 
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
        text="Distribution of Attack Types", 
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
        text="Average Active Time Distribution", 
        command=lambda: show_active_avg_distribution(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        graphical_column, 
        text="Idle Time Boxplot", 
        command=lambda: show_idle_avg_boxplot(df), 
        width=30, 
        bg="lightgray", 
        font=("Arial", 12)
    ).pack(pady=5) 

    tk.Button(
        graphical_column, 
        text="Packets vs Payload Relationship", 
        command=lambda: show_packet_vs_payload_scatter(df), 
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
