from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegressionAnalysis:
    def __init__(self, X_train, X_test, y_train, y_test, le):
        self.model = LinearRegression()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.le = le

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        y_pred_class = y_pred.round().astype(int)
        print("Linear Regression (as classification)")
        print(classification_report(
            self.y_test, y_pred_class,
            labels=range(len(self.le.classes_)),
            target_names=self.le.classes_
        ))
        cm = confusion_matrix(self.y_test, y_pred_class, labels=range(len(self.le.classes_)))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
        plt.title("Matriz de Confusão: Linear Regression")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.show()
