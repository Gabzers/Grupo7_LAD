from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class KNNAnalysis:
    def __init__(self, X_train, X_test, y_train, y_test, le, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.le = le
        self.n_neighbors = n_neighbors

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print(f"KNN (k={self.n_neighbors})")
        print(classification_report(
            self.y_test, y_pred,
            labels=range(len(self.le.classes_)),
            target_names=self.le.classes_
        ))
        cm = confusion_matrix(self.y_test, y_pred, labels=range(len(self.le.classes_)))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
        plt.title(f"Matriz de Confusão: KNN (k={self.n_neighbors})")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.show()
