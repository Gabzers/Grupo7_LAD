# Projeto de Machine Learning - Objetivo Final
# O objetivo deste projeto é a classificação dos tipos de ataques em redes IoT utilizando Random Forest.

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.decomposition import PCA, TruncatedSVD
import time
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from RandomForestAnalysis import RandomForestAnalysis

def prepare_data():
    """Prepara os dados para treinamento e teste."""
    print("Carregando e preparando dados...")
    df = pd.read_csv("RT_IOT2022.csv")
    df.drop(columns=["Unnamed: 0"], inplace=True)
    
    # Codificar colunas categóricas
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "Attack_type" in categorical_cols:
        categorical_cols.remove("Attack_type")
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Codificar target
    le = LabelEncoder()
    df["Attack_type"] = le.fit_transform(df["Attack_type"])
    
    # Separar features e target
    X = df.drop("Attack_type", axis=1)
    y = df["Attack_type"]
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le, X.columns

def run_random_forest_with_metrics(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names):
    """Executa apenas o modelo Random Forest e retorna métricas, incluindo Cross Validation."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    name = "Random Forest"
    model = RandomForestAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names=feature_names)
    print(f"\nExecutando {name}...")
    start = time.time()
    model.apply_pca()
    model.model.fit(model.X_train, model.y_train)
    y_pred = model.model.predict(model.X_test)
    end = time.time()
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} - Accuracy: {acc:.3f} | F1-score: {f1:.3f} | Tempo: {end-start:.2f}s")

    # Cross Validation (Crossfitting)
    print("\n--- Cross Validation (Random Forest) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model.model, model.X_train, model.y_train, cv=skf, scoring='accuracy')
    print(f"Cross Validation - Accuracy média: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"Scores individuais: {cv_scores}")

    return [{
        'name': name,
        'accuracy': acc,
        'f1_score': f1,
        'time': end-start,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }]

def plot_comparative_graphs(results):
    import matplotlib.pyplot as plt
    names = [r['name'] for r in results]
    accs = [r['accuracy'] for r in results]
    f1s = [r['f1_score'] for r in results]
    times = [r['time'] for r in results]
    plt.figure(figsize=(12,5))
    plt.bar(names, accs, color='royalblue')
    plt.ylabel('Accuracy')
    plt.title('Comparação de Accuracy dos Modelos')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12,5))
    plt.bar(names, f1s, color='seagreen')
    plt.ylabel('F1-score (weighted)')
    plt.title('Comparação de F1-score dos Modelos')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12,5))
    plt.bar(names, times, color='darkorange')
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Comparação de Tempo de Execução dos Modelos')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

def run_advanced_analysis(X_train_scaled, X_test_scaled, y_train, y_test, le):
    """Executa análises avançadas (PCA, SVD, Clustering, etc.)."""
    print("\nExecutando análises avançadas...")
    
    # PCA
    print("\n--- PCA ---")
    pca = PCA(n_components=10)
    start = time.time()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    end = time.time()
    print(f"PCA fit+transform time: {end-start:.2f} seconds")
    
    # SVD
    print("\n--- SVD ---")
    svd = TruncatedSVD(n_components=10)
    X_train_svd = svd.fit_transform(X_train_scaled)
    X_test_svd = svd.transform(X_test_scaled)
    
    # KMeans Clustering
    print("\n--- KMeans Clustering ---")
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_train_scaled)
        inertia.append(kmeans.inertia_)
    
    plt.figure()
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inertia')
    plt.title('Método do Cotovelo para KMeans')
    plt.show()
    
    # Hierarchical Clustering
    print("\n--- Hierarchical Clustering ---")
    plt.figure(figsize=(10, 7))
    linked = linkage(X_train_scaled[:500], 'ward')
    dendrogram(linked)
    plt.title('Dendrograma (Hierarchical Clustering)')
    plt.xlabel('Amostras')
    plt.ylabel('Distância')
    plt.show()
    
    # Cross Validation
    print("\n--- Cross Validation ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross Validation (Random Forest) - Accuracy média: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"Scores individuais dos folds: {scores}")

    # Visualização dos scores dos folds
    plt.figure(figsize=(8,4))
    plt.bar(range(1, len(scores)+1), scores, color='skyblue')
    plt.axhline(scores.mean(), color='red', linestyle='--', label=f'Média: {scores.mean():.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Scores de Accuracy por Fold no Cross Validation')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Preparar dados
    X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names = prepare_data()
    
    # Executar apenas Random Forest e obter métricas
    results = run_random_forest_with_metrics(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names)
    plot_comparative_graphs(results)
    
    # Executar análises avançadas
    run_advanced_analysis(X_train_scaled, X_test_scaled, y_train, y_test, le)