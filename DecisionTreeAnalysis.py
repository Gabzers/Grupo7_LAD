from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DecisionTreeAnalysis:
    def __init__(self, X_train, X_test, y_train, y_test, le, feature_names=None):
        self.model = DecisionTreeClassifier(random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.le = le
        self.feature_names = feature_names

    def run(self):
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
        if self.feature_names is not None:
            plt.figure(figsize=(20, 10))
            plot_tree(self.model, feature_names=self.feature_names, class_names=self.le.classes_, filled=True, max_depth=2)
            plt.title("Árvore de Decisão (max_depth=2)")
            plt.show()
        print("Gini feature importances:", self.model.feature_importances_)

if __name__ == "__main__":
    from Models import X_train_scaled, X_test_scaled, y_train, y_test, le, X
    dt_analysis = DecisionTreeAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names=X.columns)
    dt_analysis.run()

