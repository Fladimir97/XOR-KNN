import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        self.current_Learn_Iteration = 0
        # Das KNN wird mit random Werten initialisiert
        self.w1 = np.random.randn(2,4)
        self.b1 = np.zeros((1,4))
        self.w2 = np.random.randn(4,1)
        self.b2 = np.zeros((1,1))
        # Liste für die spätere Visualisierung der Loss Einträge
        self.lossHistory = []

    def forward(self, X):
        self.Z1 =  X @ self.w1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.w2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        self.dZ2 = (self.A2 - y) / X.shape[0]
        self.dW2 = self.A1.T @ self.dZ2
        self.dB2 = np.sum(self.dZ2, axis=0, keepdims=True)

        self.dA1 = self.dZ2 @ self.w2.T
        self.dZ1 = self.dA1 * self.relu_Derivative(self.Z1)

        self.dW1 = X.T @ self.dZ1
        self.dB1 = np.sum(self.dZ1, axis=0, keepdims=True)


    def update(self, learning_rate):
        self.w1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.dB1
        self.w2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.dB2

    def train(self, X, y, epochs=10000, lr=0.1):
        for i in range(epochs):
            self.forward(X)
            loss = - np.mean(y * np.log(self.A2) + (1 - y) * np.log(1 - self.A2))
            self.lossHistory.append((self.current_Learn_Iteration, loss))
            if self.current_Learn_Iteration % 1000 == 0:
                print(f"Loss nach {self.current_Learn_Iteration} Epochen:\t{loss}")
            self.backward(X, y)
            self.update(lr)
            self.current_Learn_Iteration += 1


    def relu(self, value):
        return np.maximum(0,value)

    def relu_Derivative(self, value):
        return (value > 0).astype(float)

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def visualize_Loss(self):
        x,y = zip(*self.lossHistory)
        plt.xlabel("Epoche $t$")
        plt.ylabel("Loss $L$")
        plt.title("$L(t)$")
        plt.plot(x,y)
        plt.show()

    def visualize_Boundary(self, comment="", save=False, filename="myPlot.png"):
        xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        predictions = self.forward(grid_points)
        Z = predictions.reshape(xx.shape)
        plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 50), cmap="viridis", vmin=0, vmax=1)

        plt.colorbar(label="Vorhersage $\\hat{y}$")
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,1,1,0])

        for label, marker in [(0, "o"), (1, "x")]:
            mask = (y == label)
            plt.scatter(X[mask, 0], X[mask, 1], c="white", edgecolors="black",
                        marker=marker, s=150, label=f"y={label}")

        plt.legend()
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(f"Decision Boundary {comment}")
        if save:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()



if __name__ == "__main__":
    N = NeuralNetwork()

    # Eingabe Werte - Alle Kombinationen für XOR
    X = np.array(
        [
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ]
    )

    # Ziel Werte für X
    Y = np.array(
        [
            [0],
            [1],
            [1],
            [0]
        ]
    )

    for i in range(101):
        N.visualize_Boundary(comment=f"nach $t={i*10}$ Epochen", save=True, filename=f"./images/{i:05}.png")
        N.train(X, Y, 10)

    N.visualize_Loss()
