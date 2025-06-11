from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

class RandomForestAnalysis:
    def __init__(self, X_train, X_test, y_train, y_test, le, feature_names=None):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.le = le
        self.feature_names = feature_names

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print("Random Forest")
        print(classification_report(
            self.y_test, y_pred,
            labels=range(len(self.le.classes_)),
            target_names=self.le.classes_
        ))
        cm = confusion_matrix(self.y_test, y_pred, labels=range(len(self.le.classes_)))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
        plt.title("Matriz de Confusão: Random Forest")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.show()
        if self.feature_names is not None:
            plt.figure(figsize=(20, 10))
            plot_tree(self.model.estimators_[0], feature_names=self.feature_names, class_names=self.le.classes_, filled=True, max_depth=2)
            plt.title("Uma árvore do Random Forest (max_depth=2)")
            plt.show()
