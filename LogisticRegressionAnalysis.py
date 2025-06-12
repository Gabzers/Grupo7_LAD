from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class LogisticRegressionAnalysis:
    def __init__(self, X_train, X_test, y_train, y_test, le, use_pca=True, n_components=10, use_clustering=True, n_clusters=5):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.le = le
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = None
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        
    def apply_clustering(self):
        """Aplica KMeans para análise exploratória e adiciona cluster_id como feature."""
        if self.use_clustering:
            print(f"\nAplicando KMeans Clustering com {self.n_clusters} clusters para análise exploratória e feature engineering...")
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_train = kmeans.fit_predict(self.X_train)
            cluster_test = kmeans.predict(self.X_test)
            import numpy as np
            self.X_train = np.column_stack([self.X_train, cluster_train])
            self.X_test = np.column_stack([self.X_test, cluster_test])
            print("Cluster_id adicionado como nova feature para Logistic Regression.")

    def apply_pca(self):
        """Aplica PCA aos dados de treino e teste."""
        if self.use_pca:
            print(f"\nAplicando PCA com {self.n_components} componentes...")
            self.pca = PCA(n_components=self.n_components)
            self.X_train = self.pca.fit_transform(self.X_train)
            self.X_test = self.pca.transform(self.X_test)
            print(f"Variância explicada: {sum(self.pca.explained_variance_ratio_):.3f}")
            
            # Plot da variância explicada
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1),
                    self.pca.explained_variance_ratio_.cumsum(), 'bo-')
            plt.xlabel('Número de Componentes')
            plt.ylabel('Variância Explicada Acumulada')
            plt.title('Variância Explicada pelo PCA')
            plt.grid(True)
            plt.show()

    def run(self):
        # 1. Clustering para análise exploratória e feature engineering
        self.apply_clustering()
        # 2. Aplicar PCA se necessário
        self.apply_pca()
        
        # Treinar e avaliar o modelo
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print("Logistic Regression")
        print(classification_report(
            self.y_test, y_pred,
            labels=range(len(self.le.classes_)),
            target_names=self.le.classes_
        ))
        cm = confusion_matrix(self.y_test, y_pred, labels=range(len(self.le.classes_)))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
        plt.title("Matriz de Confusão: Logistic Regression")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.show()

if __name__ == "__main__":
    # Carregar e preparar os dados
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
    
    print("Treinando modelo Logistic Regression...")
    # Instanciar e executar análise com clustering e PCA
    analysis = LogisticRegressionAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, use_pca=True, use_clustering=True)
    analysis.run()
