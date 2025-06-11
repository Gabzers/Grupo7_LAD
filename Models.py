import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

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
# ...importe outras análises conforme necessário...

if __name__ == "__main__":
    lr_analysis = LogisticRegressionAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, X.columns)
    lr_analysis.run()

    rf_analysis = RandomForestAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, X.columns)
    rf_analysis.run()

    svm_linear_analysis = SVMAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, kernel='linear')
    svm_linear_analysis.run()

    svm_rbf_analysis = SVMAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, kernel='rbf')
    svm_rbf_analysis.run()

    # ...execute outras análises conforme necessário...
