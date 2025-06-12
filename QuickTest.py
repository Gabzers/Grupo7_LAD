import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def quick_test_model(model_class, model_name, **kwargs):
    """Teste rápido de um modelo específico."""
    print(f"🧪 Teste rápido: {model_name}")
    print("-" * 40)
    
    # Preparar dados (versão simplificada)
    df = pd.read_csv("RT_IOT2022.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    
    # Usar apenas uma amostra pequena para teste rápido
    df_sample = df.sample(n=1000, random_state=42)
    
    # Codificar dados
    categorical_cols = df_sample.select_dtypes(include=["object"]).columns.tolist()
    if "Attack_type" in categorical_cols:
        categorical_cols.remove("Attack_type")
    df_sample = pd.get_dummies(df_sample, columns=categorical_cols)
    
    le = LabelEncoder()
    df_sample["Attack_type"] = le.fit_transform(df_sample["Attack_type"])
    
    X = df_sample.drop("Attack_type", axis=1)
    y = df_sample["Attack_type"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Testar modelo
    try:
        if 'feature_names' in kwargs:
            kwargs['feature_names'] = X.columns
            
        model = model_class(X_train_scaled, X_test_scaled, y_train, y_test, le, **kwargs)
        model.run()
        print("✅ Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {str(e)}")

# Exemplos de uso para testar modelos individuais
if __name__ == "__main__":
    print("🔬 TESTE RÁPIDO DE MODELOS")
    print("=" * 30)
    
    # Descomente a linha do modelo que deseja testar:
    
    # from SVMAnalysis import SVMAnalysis
    # quick_test_model(SVMAnalysis, "SVM", kernel='rbf', use_clustering=True, use_pca=True)
    
    # from DecisionTreeAnalysis import DecisionTreeAnalysis
    # quick_test_model(DecisionTreeAnalysis, "Decision Tree", use_clustering=True, use_pca=True)
    
    # from KNNAnalysis import KNNAnalysis
    # quick_test_model(KNNAnalysis, "KNN", n_neighbors=3, use_clustering=True, use_pca=True)
    
    from RandomForestAnalysis import RandomForestAnalysis
    quick_test_model(RandomForestAnalysis, "Random Forest", use_clustering=True, use_pca=True)
