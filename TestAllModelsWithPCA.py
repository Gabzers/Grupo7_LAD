import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
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

class ModelTesterWithPCA:
    def __init__(self, csv_file="RT_IOT2022.csv", n_components=0.95):
        self.csv_file = csv_file
        self.n_components = n_components  # Manter 95% da variância
        self.results = {}
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train_pca = None
        self.X_test_pca = None
        self.y_train = None
        self.y_test = None
        self.le = None
        self.feature_names = None
        self.pca = None
        self.pca_feature_names = None
        
    def prepare_data(self):
        """Prepara os dados para todos os modelos COM PCA aplicado globalmente."""
        print("🔄 Preparando dados COM PCA...")
        
        # Carregar dados
        df = pd.read_csv(self.csv_file)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        
        # Codificar colunas categóricas
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if "Attack_type" in categorical_cols:
            categorical_cols.remove("Attack_type")
        df = pd.get_dummies(df, columns=categorical_cols)
        
        # Codificar target
        self.le = LabelEncoder()
        df["Attack_type"] = self.le.fit_transform(df["Attack_type"])
        
        # Separar features e target
        X = df.drop("Attack_type", axis=1)
        y = df["Attack_type"]
        self.feature_names = X.columns
        
        print(f"📊 Features originais: {X.shape[1]}")
        
        # Split treino/teste
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Normalização ANTES do PCA
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        
        # 🔄 APLICAR PCA AQUI!
        print(f"🔄 Aplicando PCA com {self.n_components} de variância...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)
        
        # Gerar nomes para as componentes PCA
        n_components_final = self.X_train_pca.shape[1]
        self.pca_feature_names = [f"PC{i+1}" for i in range(n_components_final)]
        
        print(f"✅ PCA aplicado: {X.shape[1]} → {n_components_final} componentes")
        print(f"📈 Variância explicada total: {self.pca.explained_variance_ratio_.sum():.4f}")
        print(f"✅ Dados preparados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
        
    def test_model(self, model_class, model_name, **kwargs):
        """Testa um modelo específico com dados PCA e retorna métricas."""
        print(f"\n🧪 Testando {model_name} (COM PCA)...")
        start_time = time.time()
        
        try:
            # Usar dados COM PCA e nomes das componentes PCA
            if 'feature_names' in kwargs:
                kwargs['feature_names'] = self.pca_feature_names
            
            # ⚠️ DESABILITAR PCA interno do modelo já que aplicamos globalmente
            if 'use_pca' in kwargs:
                kwargs['use_pca'] = False
                
            model = model_class(
                self.X_train_pca.copy(), 
                self.X_test_pca.copy(),
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
                'n_features': self.X_train_pca.shape[1],
                'status': 'Success'
            }
            
            print(f"✅ {model_name} concluído em {execution_time:.2f}s")
            print(f"   Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
            print(f"   Features: {self.X_train_pca.shape[1]} (PCA)")
            
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
                'n_features': self.X_train_pca.shape[1],
                'status': f'Error: {str(e)}'
            }
    
    def run_all_tests(self):
        """Executa todos os modelos com dados PCA."""
        print("🚀 Iniciando testes de todos os modelos COM PCA...")
        
        # Configurações de teste (PCA já aplicado globalmente)
        test_configs = [
            (SVMAnalysis, "SVM (RBF)", {'kernel': 'rbf', 'use_pca': False, 'use_clustering': True}),
            (DecisionTreeAnalysis, "Decision Tree", {'use_pca': False, 'use_clustering': True}),
            (RandomForestAnalysis, "Random Forest", {'use_pca': False, 'use_clustering': True}),
            (KNNAnalysis, "KNN", {'n_neighbors': 5, 'use_pca': False, 'use_clustering': True}),
            (LogisticRegressionAnalysis, "Logistic Regression", {'use_pca': False, 'use_clustering': True}),
            (MLPAnalysis, "Neural Network", {'hidden_layer_sizes': (50,), 'use_pca': False, 'use_clustering': True}),
            (NaiveBayesAnalysis, "Naive Bayes", {'use_pca': False, 'use_clustering': True}),
            (LassoAnalysis, "Lasso Regression", {'alpha': 0.1, 'use_pca': False, 'use_clustering': True}),
            (LinearRegressionAnalysis, "Linear Regression", {'use_pca': False, 'use_clustering': True}),
            (RidgeAnalysis, "Ridge Regression", {'alpha': 1.0, 'use_pca': False, 'use_clustering': True})
        ]
        
        for model_class, model_name, config in test_configs:
            result = self.test_model(model_class, model_name, **config)
            self.results[model_name] = result
            
        print("\n🎉 Todos os testes concluídos!")
        
    def analyze_pca_impact(self):
        """Analisa o impacto do PCA nos dados."""
        print("\n📊 ANÁLISE DO IMPACTO DO PCA")
        print("=" * 60)
        
        # Informações sobre componentes principais
        n_components = len(self.pca.explained_variance_ratio_)
        total_variance = self.pca.explained_variance_ratio_.sum()
        
        print(f"🔢 Componentes principais: {n_components}")
        print(f"📈 Variância total explicada: {total_variance:.4f} ({total_variance*100:.2f}%)")
        print(f"📉 Redução de dimensionalidade: {len(self.feature_names)} → {n_components}")
        print(f"💾 Compressão: {(1 - n_components/len(self.feature_names))*100:.1f}%")
        
        # Top 10 componentes com maior variância
        print(f"\n🏆 Top 10 Componentes Principais:")
        for i in range(min(10, n_components)):
            print(f"   PC{i+1}: {self.pca.explained_variance_ratio_[i]:.4f} ({self.pca.explained_variance_ratio_[i]*100:.2f}%)")
        
        # Variância cumulativa
        cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
        print(f"\n📊 Variância Cumulativa:")
        milestones = [0.8, 0.9, 0.95, 0.99]
        for milestone in milestones:
            idx = np.where(cumsum_variance >= milestone)[0]
            if len(idx) > 0:
                print(f"   {milestone*100:.0f}% da variância: {idx[0]+1} componentes")
        
    def generate_report(self):
        """Gera relatório comparativo dos resultados COM PCA."""
        print("\n📊 RELATÓRIO DE RESULTADOS (COM PCA)")
        print("=" * 90)
        
        # Criar DataFrame com resultados
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.sort_values('f1_score', ascending=False)
        
        # Imprimir tabela de resultados
        print(f"{'Modelo':<20} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10} {'Features':<10} {'Tempo(s)':<10} {'Status':<15}")
        print("-" * 90)
        
        for model_name, row in df_results.iterrows():
            print(f"{model_name:<20} {row['accuracy']:<10.4f} {row['precision']:<11.4f} {row['recall']:<10.4f} {row['f1_score']:<10.4f} {row['n_features']:<10} {row['execution_time']:<10.2f} {row['status']:<15}")
        
        # Encontrar melhor modelo
        best_model = df_results.index[0]
        print(f"\n🏆 Melhor modelo COM PCA (F1-Score): {best_model}")
        print(f"   F1-Score: {df_results.loc[best_model, 'f1_score']:.4f}")
        print(f"   Accuracy: {df_results.loc[best_model, 'accuracy']:.4f}")
        print(f"   Features: {df_results.loc[best_model, 'n_features']}")
        
        # Salvar resultados
        df_results.to_csv("model_comparison_results_WITH_PCA.csv")
        print(f"\n💾 Resultados salvos em 'model_comparison_results_WITH_PCA.csv'")
        
        # Gerar gráficos comparativos
        self.plot_comparison(df_results)
        self.plot_pca_analysis()
        
    def plot_comparison(self, df_results):
        """Gera gráficos comparativos dos modelos COM PCA."""
        plt.style.use('default')
        
        # Gráfico de barras das métricas
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparação de Modelos COM PCA', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        
        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[i//2, i%2]
            df_results[metric].plot(kind='bar', ax=ax, color=color, alpha=0.8)
            ax.set_title(f'{title} por Modelo (COM PCA)', fontsize=14, fontweight='bold')
            ax.set_ylabel(title)
            ax.set_xlabel('Modelos')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for j, v in enumerate(df_results[metric]):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_comparison_metrics_WITH_PCA.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Gráfico de tempo de execução
        plt.figure(figsize=(12, 6))
        df_results['execution_time'].plot(kind='bar', color='mediumpurple', alpha=0.8)
        plt.title('Tempo de Execução por Modelo (COM PCA)', fontsize=14, fontweight='bold')
        plt.ylabel('Tempo (segundos)')
        plt.xlabel('Modelos')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(df_results['execution_time']):
            plt.text(i, v + 0.5, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        plt.savefig('model_execution_times_WITH_PCA.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_analysis(self):
        """Gera gráficos de análise do PCA."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico 1: Variância explicada por componente
        n_components = min(20, len(self.pca.explained_variance_ratio_))
        components = range(1, n_components + 1)
        variance_ratios = self.pca.explained_variance_ratio_[:n_components]
        
        axes[0].bar(components, variance_ratios, alpha=0.8, color='steelblue')
        axes[0].set_title('Variância Explicada por Componente Principal', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Componente Principal')
        axes[0].set_ylabel('Variância Explicada')
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Variância cumulativa
        cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'o-', color='darkorange', linewidth=2)
        axes[1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% da variância')
        axes[1].set_title('Variância Cumulativa Explicada', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Número de Componentes')
        axes[1].set_ylabel('Variância Cumulativa')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Função principal para executar todos os testes COM PCA."""
    print("🔬 SISTEMA DE TESTE DE MODELOS COM PCA GLOBAL")
    print("=" * 55)
    
    # Criar tester
    tester = ModelTesterWithPCA(n_components=0.95)  # Manter 95% da variância
    
    # Preparar dados COM PCA
    tester.prepare_data()
    
    # Analisar impacto do PCA
    tester.analyze_pca_impact()
    
    # Executar todos os testes
    tester.run_all_tests()
    
    # Gerar relatório
    tester.generate_report()
    
    print("\n✨ Análise completa COM PCA finalizada!")
    print("📁 Arquivos gerados:")
    print("   - model_comparison_results_WITH_PCA.csv")
    print("   - model_comparison_metrics_WITH_PCA.png")
    print("   - model_execution_times_WITH_PCA.png")
    print("   - pca_analysis.png")

if __name__ == "__main__":
    main()
