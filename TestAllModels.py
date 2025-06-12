import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')

# Import all analysis classes
from SVMAnalysis import SVMAnalysis
from DecisionTreeAnalysis import DecisionTreeAnalysis
from KNNAnalysis import KNNAnalysis
from LogisticRegressionAnalysis import LogisticRegressionAnalysis
from MLPAnalysis import MLPAnalysis
from NaiveBayesAnalysis import NaiveBayesAnalysis
from LassoAnalysis import LassoAnalysis
from LinearRegressionAnalysis import LinearRegressionAnalysis
from RidgeAnalysis import RidgeAnalysis
from RandomForestAnalysis import RandomForestAnalysis

class ModelTester:
    def __init__(self, csv_file="RT_IOT2022.csv"):
        self.csv_file = csv_file
        self.results = {}
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.le = None
        self.feature_names = None
        
    def prepare_data(self):
        """Prepara os dados para todos os modelos."""
        print("🔄 Preparando dados...")
        
        # Carregar dados
        df = pd.read_csv(self.csv_file)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        
        # Codificar colunas categóricas (ANTES do PCA)
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if "Attack_type" in categorical_cols:
            categorical_cols.remove("Attack_type")
        df = pd.get_dummies(df, columns=categorical_cols)
        
        # Codificar target (ANTES do PCA)
        self.le = LabelEncoder()
        df["Attack_type"] = self.le.fit_transform(df["Attack_type"])
        
        # Separar features e target (ANTES do PCA)
        X = df.drop("Attack_type", axis=1)
        y = df["Attack_type"]
        self.feature_names = X.columns
        
        # Split treino/teste (ANTES do PCA)
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Normalização (ANTES do PCA)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        
        # ⚠️ PCA NÃO É APLICADO AQUI!
        
        print(f"✅ Dados preparados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
        print(f"📊 Features: {X_train.shape[1]}, Classes: {len(self.le.classes_)}")
        
    def test_model(self, model_class, model_name, **kwargs):
        """Testa um modelo específico e retorna métricas."""
        print(f"\n🧪 Testando {model_name}...")
        start_time = time.time()
        
        try:
            # Criar e executar modelo
            if 'feature_names' in kwargs:
                kwargs['feature_names'] = self.feature_names
                
            model = model_class(
                self.X_train_scaled.copy(), 
                self.X_test_scaled.copy(),
                self.y_train.copy(), 
                self.y_test.copy(), 
                self.le,
                **kwargs
            )
            
            # Capturar stdout para evitar prints excessivos
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            model.run()
            
            # Restaurar stdout
            sys.stdout = old_stdout
            output = mystdout.getvalue()
            
            # Calcular métricas
            y_pred = model.model.predict(model.X_test)
            if hasattr(model.model, 'predict_proba'):
                # Para modelos de classificação
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            else:
                # Para modelos de regressão usados como classificação
                y_pred_class = y_pred.round().astype(int)
                # Garantir que as predições estão no range válido
                y_pred_class = np.clip(y_pred_class, 0, len(self.le.classes_)-1)
                accuracy = accuracy_score(self.y_test, y_pred_class)
                precision = precision_score(self.y_test, y_pred_class, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred_class, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred_class, average='weighted', zero_division=0)
            
            execution_time = time.time() - start_time
            
            result = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'execution_time': execution_time,
                'status': 'Success'
            }
            
            print(f"✅ {model_name} concluído em {execution_time:.2f}s")
            print(f"   Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Erro em {model_name}: {str(e)}")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'execution_time': execution_time,
                'status': f'Error: {str(e)}'
            }
    
    def run_all_tests(self):
        """Executa todos os modelos e coleta resultados."""
        print("🚀 Iniciando testes de todos os modelos...")
        
        # Configurações de teste
        test_configs = [
            # ⚠️ PCA É APLICADO AQUI, DENTRO DE CADA MODELO!
            (SVMAnalysis, "SVM (RBF)", {'kernel': 'rbf', 'use_pca': True, 'use_clustering': True}),
            (DecisionTreeAnalysis, "Decision Tree", {'use_pca': True, 'use_clustering': True}),
            (RandomForestAnalysis, "Random Forest", {'use_pca': True, 'use_clustering': True}),
            (KNNAnalysis, "KNN", {'n_neighbors': 5, 'use_pca': True, 'use_clustering': True}),
            (LogisticRegressionAnalysis, "Logistic Regression", {'use_pca': True, 'use_clustering': True}),
            (MLPAnalysis, "Neural Network", {'hidden_layer_sizes': (50,), 'use_pca': True, 'use_clustering': True}),
            (NaiveBayesAnalysis, "Naive Bayes", {'use_pca': True, 'use_clustering': True}),
            (LassoAnalysis, "Lasso Regression", {'alpha': 0.1, 'use_pca': True, 'use_clustering': True}),
            (LinearRegressionAnalysis, "Linear Regression", {'use_pca': True, 'use_clustering': True}),
            (RidgeAnalysis, "Ridge Regression", {'alpha': 1.0, 'use_pca': True, 'use_clustering': True})
        ]
        
        for model_class, model_name, config in test_configs:
            result = self.test_model(model_class, model_name, **config)
            self.results[model_name] = result
            
        print("\n🎉 Todos os testes concluídos!")
        
    def generate_report(self):
        """Gera relatório comparativo dos resultados."""
        print("\n📊 RELATÓRIO DE RESULTADOS")
        print("=" * 80)
        
        # Criar DataFrame com resultados
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.sort_values('f1_score', ascending=False)
        
        # Imprimir tabela de resultados
        print(f"{'Modelo':<20} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10} {'Tempo(s)':<10} {'Status':<15}")
        print("-" * 80)
        
        for model_name, row in df_results.iterrows():
            print(f"{model_name:<20} {row['accuracy']:<10.4f} {row['precision']:<11.4f} {row['recall']:<10.4f} {row['f1_score']:<10.4f} {row['execution_time']:<10.2f} {row['status']:<15}")
        
        # Encontrar melhor modelo
        best_model = df_results.index[0]
        print(f"\n🏆 Melhor modelo (F1-Score): {best_model}")
        print(f"   F1-Score: {df_results.loc[best_model, 'f1_score']:.4f}")
        print(f"   Accuracy: {df_results.loc[best_model, 'accuracy']:.4f}")
        
        # Salvar resultados
        df_results.to_csv("model_comparison_results.csv")
        print(f"\n💾 Resultados salvos em 'model_comparison_results.csv'")
        
        # Gerar gráficos comparativos
        self.plot_comparison(df_results)
        
    def plot_comparison(self, df_results):
        """Gera gráficos comparativos dos modelos."""
        plt.style.use('default')
        
        # Gráfico de barras das métricas
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            df_results[metric].plot(kind='bar', ax=ax, color='skyblue', alpha=0.8)
            ax.set_title(f'{title} por Modelo', fontsize=14, fontweight='bold')
            ax.set_ylabel(title)
            ax.set_xlabel('Modelos')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for j, v in enumerate(df_results[metric]):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Gráfico de tempo de execução
        plt.figure(figsize=(12, 6))
        df_results['execution_time'].plot(kind='bar', color='lightcoral', alpha=0.8)
        plt.title('Tempo de Execução por Modelo', fontsize=14, fontweight='bold')
        plt.ylabel('Tempo (segundos)')
        plt.xlabel('Modelos')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(df_results['execution_time']):
            plt.text(i, v + 0.5, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        plt.savefig('model_execution_times.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Função principal para executar todos os testes."""
    print("🔬 SISTEMA DE TESTE DE MODELOS COM CLUSTERING")
    print("=" * 50)
    
    # Criar tester
    tester = ModelTester()
    
    # Preparar dados
    tester.prepare_data()
    
    # Executar todos os testes
    tester.run_all_tests()
    
    # Gerar relatório
    tester.generate_report()
    
    print("\n✨ Análise completa finalizada!")
    print("📁 Arquivos gerados:")
    print("   - model_comparison_results.csv")
    print("   - model_comparison_metrics.png")
    print("   - model_execution_times.png")

if __name__ == "__main__":
    main()
