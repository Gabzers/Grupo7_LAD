import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.decomposition import PCA, TruncatedSVD
import time
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Carregamento do dataset
df = pd.read_csv("RT_IOT2022.csv")

# 2. Remover colunas desnecessárias
df.drop(columns=["Unnamed: 0"], inplace=True)

# 3. Codificar colunas categóricas (exceto target)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "Attack_type" in categorical_cols:
    categorical_cols.remove("Attack_type")
df = pd.get_dummies(df, columns=categorical_cols)

# 4. Codificar variável alvo
le = LabelEncoder()
df["Attack_type"] = le.fit_transform(df["Attack_type"])

# 5. Separar features e target
X = df.drop("Attack_type", axis=1)
y = df["Attack_type"]

# 6. Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 7. Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Importar as classes de análise
from LogisticRegressionAnalysis import LogisticRegressionAnalysis
from RandomForestAnalysis import RandomForestAnalysis
from SVMAnalysis import SVMAnalysis
from DecisionTreeAnalysis import DecisionTreeAnalysis
from NaiveBayesAnalysis import NaiveBayesAnalysis
from LinearRegressionAnalysis import LinearRegressionAnalysis
from LassoAnalysis import LassoAnalysis
from RidgeAnalysis import RidgeAnalysis
from KNNAnalysis import KNNAnalysis
from MLPAnalysis import MLPAnalysis

if __name__ == "__main__":
    # Executar cada modelo
    lr_analysis = LogisticRegressionAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    lr_analysis.run()

    rf_analysis = RandomForestAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, X.columns)
    rf_analysis.run()

    svm_linear_analysis = SVMAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, kernel='linear')
    svm_linear_analysis.run()

    svm_rbf_analysis = SVMAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, kernel='rbf')
    svm_rbf_analysis.run()

    dt_analysis = DecisionTreeAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, X.columns)
    dt_analysis.run()

    nb_analysis = NaiveBayesAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    nb_analysis.run()

    linreg_analysis = LinearRegressionAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    linreg_analysis.run()

    lasso_analysis = LassoAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    lasso_analysis.run()

    ridge_analysis = RidgeAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    ridge_analysis.run()

    knn_analysis = KNNAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    knn_analysis.run()

    mlp_analysis = MLPAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, hidden_layer_sizes=(50,), name="Neural Net (1 layer)")
    mlp_analysis.run()
    mlp_multi_analysis = MLPAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, hidden_layer_sizes=(100, 50), name="Neural Net (multi layer)")
    mlp_multi_analysis.run()

    # --- PCA ---
    pca = PCA(n_components=10)
    start = time.time()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    end = time.time()
    print(f"PCA fit+transform time: {end-start:.2f} seconds")
    rf_pca = RandomForestAnalysis(X_train_pca, X_test_pca, y_train, y_test, le)
    rf_pca.run()

    # --- SVD ---
    svd = TruncatedSVD(n_components=10)
    X_train_svd = svd.fit_transform(X_train_scaled)
    X_test_svd = svd.transform(X_test_scaled)
    rf_svd = RandomForestAnalysis(X_train_svd, X_test_svd, y_train, y_test, le)
    rf_svd.run()

    # --- KMeans Clustering ---
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

    # --- Hierarchical Clustering ---
    plt.figure(figsize=(10, 7))
    linked = linkage(X_train_scaled[:500], 'ward')
    dendrogram(linked)
    plt.title('Dendrograma (Hierarchical Clustering)')
    plt.xlabel('Amostras')
    plt.ylabel('Distância')
    plt.show()

    # --- Cross Validation ---
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross Validation (Random Forest) - Accuracy média: {scores.mean():.3f} (+/- {scores.std():.3f})")