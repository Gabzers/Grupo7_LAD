from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class RidgeAnalysis:
    def __init__(self, X_train, X_test, y_train, y_test, le, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.le = le
        self.alpha = alpha

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        y_pred_class = y_pred.round().astype(int)
        print(f"Ridge Regression (alpha={self.alpha}) as classification")
        print(classification_report(
            self.y_test, y_pred_class,
            labels=range(len(self.le.classes_)),
            target_names=self.le.classes_
        ))
        cm = confusion_matrix(self.y_test, y_pred_class, labels=range(len(self.le.classes_)))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
        plt.title(f"Matriz de Confusão: Ridge Regression (alpha={self.alpha})")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.show()
