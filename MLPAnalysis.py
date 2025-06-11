from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLPAnalysis:
    def __init__(self, X_train, X_test, y_train, y_test, le, hidden_layer_sizes=(50,), name="Neural Net"):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.le = le
        self.name = name

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print(self.name)
        print(classification_report(
            self.y_test, y_pred,
            labels=range(len(self.le.classes_)),
            target_names=self.le.classes_
        ))
        cm = confusion_matrix(self.y_test, y_pred, labels=range(len(self.le.classes_)))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
        plt.title(f"Matriz de Confusão: {self.name}")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.show()
