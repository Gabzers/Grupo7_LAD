from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

class DecisionTreeAnalysis:
    def __init__(self, X_train, X_test, y_train, y_test, le, feature_names=None, use_pca=True, n_components=10):
        self.model = DecisionTreeClassifier(random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.le = le
        self.feature_names = feature_names
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = None
    
    def apply_pca(self):
        if self.use_pca:
            print(f"\nAplicando PCA com {self.n_components} componentes...")
            self.pca = PCA(n_components=self.n_components)
            self.X_train = self.pca.fit_transform(self.X_train)
            self.X_test = self.pca.transform(self.X_test)
            print(f"Variância explicada: {sum(self.pca.explained_variance_ratio_):.3f}")
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1),
                    self.pca.explained_variance_ratio_.cumsum(), 'bo-')
            plt.xlabel('Número de Componentes')
            plt.ylabel('Variância Explicada Acumulada')
            plt.title('Variância Explicada pelo PCA')
            plt.grid(True)
            plt.show()
    def run(self):
        self.apply_pca()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print("Decision Tree")
        print(classification_report(
            self.y_test, y_pred,
            labels=range(len(self.le.classes_)),
            target_names=self.le.classes_
        ))
        cm = confusion_matrix(self.y_test, y_pred, labels=range(len(self.le.classes_)))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
        plt.title("Matriz de Confusão: Decision Tree")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.show()
        if self.feature_names is not None and not self.use_pca:
            plt.figure(figsize=(20, 10))
            plot_tree(self.model, feature_names=self.feature_names, class_names=self.le.classes_, filled=True, max_depth=2)
            plt.title("Árvore de Decisão (max_depth=2)")
        print("Gini feature importances:", self.model.feature_importances_)

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
    
    print("Treinando modelo Decision Tree...")
    # Instanciar e executar análise
    analysis = DecisionTreeAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names=X.columns, use_pca=True)
    analysis.run()

