# Projeto de Machine Learning - Objetivo Final
# O objetivo deste projeto é a classificação dos tipos de ataques em redes IoT utilizando diferentes algoritmos de machine learning.
# O modelo prevê a classe (tipo de ataque) para cada instância do dataset.

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
import matplotlib.pyplot as plt
from sklearn.svm import SVC

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

# --- KMeans Clustering: Análise do número ótimo de clusters ---
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

# Escolher o número ótimo de clusters (por exemplo, k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)
print(f"KMeans - Labels dos clusters (k={optimal_k}):", clusters[:10])

# --- Hierarchical Clustering: Dendrograma ---
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(10, 7))
linked = linkage(X_train_scaled[:500], 'ward')
dendrogram(linked)
plt.title('Dendrograma (Hierarchical Clustering)')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()

# --- Cross Validation ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross Validation (Random Forest) - Accuracy média: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Exemplo de análise do fit time para cada modelo
if __name__ == "__main__":
    # Logistic Regression
    start = time.time()
    lr = LogisticRegressionAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    lr.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"Logistic Regression fit time: {end-start:.3f} seconds")

    # Random Forest
    start = time.time()
    rf = RandomForestAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    rf.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"Random Forest fit time: {end-start:.3f} seconds")

    # SVM (linear)
    start = time.time()
    svm_linear = SVMAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, kernel='linear')
    svm_linear.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"SVM (linear) fit time: {end-start:.3f} seconds")

    # SVM (rbf)
    start = time.time()
    svm_rbf = SVMAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, kernel='rbf')
    svm_rbf.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"SVM (rbf) fit time: {end-start:.3f} seconds")

    # Decision Tree
    start = time.time()
    dt = DecisionTreeAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names=X.columns)
    dt.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"Decision Tree fit time: {end-start:.3f} seconds")

    # KNN
    start = time.time()
    knn = KNNAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    knn.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"KNN fit time: {end-start:.3f} seconds")

    # Naive Bayes
    start = time.time()
    nb = NaiveBayesAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    nb.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"Naive Bayes fit time: {end-start:.3f} seconds")

    # MLP
    start = time.time()
    mlp = MLPAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    mlp.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"MLP fit time: {end-start:.3f} seconds")

    # Lasso
    start = time.time()
    lasso = LassoAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    lasso.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"Lasso fit time: {end-start:.3f} seconds")

    # Ridge
    start = time.time()
    ridge = RidgeAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le)
    ridge.model.fit(X_train_scaled, y_train)
    end = time.time()
    print(f"Ridge fit time: {end-start:.3f} seconds")

# --- Comparação de Modelos: Dados Originais, PCA e SVD ---

def evaluate_model(model, X_train, X_test, y_train, y_test, desc=""):
    start = time.time()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    end = time.time()
    print(f"{desc} - Accuracy: {acc:.4f} | Tempo de execução: {end-start:.2f} segundos")
    return acc, end-start

print("\n--- Comparação de Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf, X_train_scaled, X_test_scaled, y_train, y_test, desc="Random Forest (Original)")

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
evaluate_model(rf, X_train_pca, X_test_pca, y_train, y_test, desc="Random Forest (PCA)")

# SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=10)
X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd = svd.transform(X_test_scaled)
evaluate_model(rf, X_train_svd, X_test_svd, y_train, y_test, desc="Random Forest (SVD)")

print("\n--- Comparação de SVM ---")
svm = SVC(kernel='rbf', random_state=42)
evaluate_model(svm, X_train_scaled, X_test_scaled, y_train, y_test, desc="SVM (Original)")
evaluate_model(svm, X_train_pca, X_test_pca, y_train, y_test, desc="SVM (PCA)")
evaluate_model(svm, X_train_svd, X_test_svd, y_train, y_test, desc="SVM (SVD)")

# Treinar modelo para obter métricas
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

# Relatório de classificação (precision, recall, f1-score)
report = classification_report(y_test, y_pred, output_dict=True)
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred))

# Gráfico: F1-score por classe
f1_scores = [report[str(i)]['f1-score'] for i in range(len(report)-3)]
plt.figure()
plt.bar(range(len(f1_scores)), f1_scores)
plt.xlabel('Classe')
plt.ylabel('F1-score')
plt.title('F1-score por Classe (Random Forest)')
plt.show()

# Gráfico: Matriz de Confusão
ConfusionMatrixDisplay.from_estimator(rf, X_test_scaled, y_test, cmap="Blues")
plt.title("Matriz de Confusão (Random Forest)")
plt.show()

# --- Aplicação de Classificação/Predictiva para o Dashboard ---
# Exemplo usando o melhor algoritmo identificado (Random Forest)

def predict_attack_type(input_data):
    """
    Recebe um array numpy com os dados de entrada já normalizados e retorna a classe prevista.
    """
    # O modelo deve ser treinado previamente
    pred = rf.predict(input_data)
    return le.inverse_transform(pred)

# Exemplo de uso:
# input_sample = X_test_scaled[0].reshape(1, -1)
# predicted_class = predict_attack_type(input_sample)
# print("Classe prevista para a amostra:", predicted_class[0])