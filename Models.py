import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
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

# 8. Inicializar modelos
models = {
    "Linear Regression": LinearRegression(),  # Para regressão, apenas para demonstração
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Ridge Regression (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso Regression (alpha=0.1)": Lasso(alpha=0.1),
    "Naive Bayes": GaussianNB(),
    "SVM (linear kernel)": SVC(kernel='linear'),
    "SVM (rbf kernel)": SVC(kernel='rbf'),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Neural Net (1 layer)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
    "Neural Net (multi layer)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# 9. Treinar e avaliar
for name, model in models.items():
    print(f"\n🔍 Modelo: {name}")
    # Linear, Ridge, Lasso são modelos de regressão, tratá-los separadamente
    if name.startswith("Linear Regression") or "Ridge" in name or "Lasso" in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # Para regressão, converter para classificação (apenas para comparação)
        y_pred_class = y_pred.round().astype(int)
        print(classification_report(
            y_test, y_pred_class, 
            labels=range(len(le.classes_)), 
            target_names=le.classes_
        ))
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print(classification_report(
            y_test, y_pred, 
            labels=range(len(le.classes_)), 
            target_names=le.classes_
        ))
    
    # Matriz de Confusão (exceto para regressão)
    if not (name.startswith("Linear Regression") or "Ridge" in name or "Lasso" in name):
        cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
        plt.title(f"Matriz de Confusão: {name}")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.show()
    
    # Decision Tree: desenhar árvore e analisar gini
    if name == "Decision Tree":
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=X.columns, class_names=le.classes_, filled=True, max_depth=2)
        plt.title("Árvore de Decisão (max_depth=2)")
        plt.show()
        print("Gini feature importances:", model.feature_importances_)
    
    # Random Forest: mostrar uma árvore
    if name == "Random Forest":
        plt.figure(figsize=(20, 10))
        plot_tree(model.estimators_[0], feature_names=X.columns, class_names=le.classes_, filled=True, max_depth=2)
        plt.title("Uma árvore do Random Forest (max_depth=2)")
        plt.show()

# Ridge/Lasso alpha analysis
alphas = [0.01, 0.1, 1, 10]
ridge_scores = []
lasso_scores = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    ridge_scores.append(ridge.score(X_test_scaled, y_test))
    lasso_scores.append(lasso.score(X_test_scaled, y_test))
plt.figure()
plt.plot(alphas, ridge_scores, label="Ridge")
plt.plot(alphas, lasso_scores, label="Lasso")
plt.xscale('log')
plt.xlabel("Alpha")
plt.ylabel("Score")
plt.title("Ridge/Lasso Alpha Analysis")
plt.legend()
plt.show()

# KNN: análise do número ótimo de vizinhos
knn_scores = []
neighbors_range = range(1, 21)
for k in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    knn_scores.append(knn.score(X_test_scaled, y_test))
plt.figure()
plt.plot(neighbors_range, knn_scores, marker='o')
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia")
plt.title("Análise do número ótimo de vizinhos para KNN")
plt.show()

# O classification report mostra várias métricas importantes para avaliar o desempenho do modelo:
# - precision: Proporção de predições positivas corretas para cada classe.
# - recall: Proporção de exemplos positivos corretamente identificados para cada classe.
# - f1-score: Média harmônica entre precision e recall (quanto mais próximo de 1, melhor).
# - support: Número de exemplos reais de cada classe no conjunto de teste.

# Para saber se o modelo é bom:
# - Olhe para os valores de precision, recall e f1-score (quanto mais próximos de 1, melhor).
# - Veja a acurácia geral (accuracy).
# - Compare macro avg (média simples entre classes) e weighted avg (média ponderada pelo número de exemplos).

# No seu caso, os valores estão muito próximos de 1, indicando que o modelo está com desempenho excelente.
# Se os valores fossem baixos (ex: <0.7), o modelo não seria considerado bom.

# Ridge Regression (alpha=1.0) não é muito bom para este problema.
# Isso porque Ridge é um modelo de regressão, não de classificação.
# Os resultados mostram f1-score baixo para várias classes e macro avg baixo (~0.32).
# Modelos de classificação (Logistic Regression, Random Forest, SVM, etc.) são mais adequados para este tipo de tarefa.

# Após treinar o modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
dump(rf_model, "random_forest_model.joblib")

# 1. Carregar o modelo treinado
model = load("random_forest_model.joblib")

# 2. Carregar os dados novos (ou o mesmo dataset para teste)
df = pd.read_csv("RT_IOT2022.csv")

# 3. Pré-processamento igual ao treino
# Remover colunas desnecessárias
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Codificar colunas categóricas (exceto target)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "Attack_type" in categorical_cols:
    categorical_cols.remove("Attack_type")
df = pd.get_dummies(df, columns=categorical_cols)

# Codificar variável alvo (para comparar, se existir)
le = LabelEncoder()
if "Attack_type" in df.columns:
    df["Attack_type"] = le.fit_transform(df["Attack_type"])
    X = df.drop("Attack_type", axis=1)
else:
    X = df

# Normalização (usa os mesmos parâmetros do treino)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Se possível, use scaler salvo do treino

# 4. Fazer a predição
predicoes = model.predict(X_scaled)

# 5. Adicionar resultados ao DataFrame
df["Predicted_Attack"] = le.inverse_transform(predicoes)

# 6. Salvar resultados
df.to_csv("resultados_com_predicoes.csv", index=False)

# 7. Exibir resumo
print(df[["Predicted_Attack"]].value_counts())

# PCA
pca = PCA(n_components=10)  # Reduz para 10 componentes principais
start = time.time()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
end = time.time()
print(f"PCA fit+transform time: {end-start:.2f} seconds")

# Exemplo: Treinar Random Forest com dados reduzidos
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
print("Random Forest com PCA:")
print(classification_report(y_test, y_pred_pca, labels=range(len(le.classes_)), target_names=le.classes_))

# SVD (TruncatedSVD)
svd = TruncatedSVD(n_components=10)
X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd = svd.transform(X_test_scaled)
rf_svd = RandomForestClassifier(n_estimators=100, random_state=42)
rf_svd.fit(X_train_svd, y_train)
y_pred_svd = rf_svd.predict(X_test_svd)
print("Random Forest com SVD:")
print(classification_report(y_test, y_pred_svd, labels=range(len(le.classes_)), target_names=le.classes_))

# KMeans: Encontrar o número ótimo de clusters (cotovelo)
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

# Hierarchical Clustering: Dendrograma
plt.figure(figsize=(10, 7))
linked = linkage(X_train_scaled[:500], 'ward')  # Usa só 500 amostras para não pesar
dendrogram(linked)
plt.title('Dendrograma (Hierarchical Clustering)')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()

# Exemplo com Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross Validation (Random Forest) - Accuracy média: {scores.mean():.3f} (+/- {scores.std():.3f})")
