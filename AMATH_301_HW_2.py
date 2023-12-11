import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# Problem 1

def factorial(n):
    factorial = 1
    for i in range (1, n + 1):
        factorial = factorial * i
    return factorial

A1 = factorial(3)
A2 = factorial(15)
A3 = factorial(23)

# Problem 2

alpha = -0.003
omega = 0.02
x_0 = np.array(([1], [-1]))
A = np.array(([1 - alpha, -omega], [omega, 1 - alpha]))
x_coord = np.zeros([1, 1001])
y_coord = np.zeros([1, 1001])
for i in range (0, 1001):
    x_i = np.linalg.matrix_power(np.linalg.inv(A), i) @ x_0
    x_coord[0][i] = x_i[0][0]
    y_coord[0][i] = x_i[1][0]

A4 = x_coord
A5 = y_coord

distances = np.zeros([1, 1001])
for i in range (0, 1001):
    distances[0][i] = np.sqrt(x_coord[0][i] ** 2 + y_coord[0][i] ** 2)

A6 = distances

plt.scatter(x_coord, y_coord)
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.title("XY-Positions of Oscillatory Particle Over Time")
plt.show()

# Problem 3

n = 20
A = np.zeros([n, n])
for i in range (n):
    for j in range (n):
        if (i == j):
            A[i][j] = 10 * np.sin(i)
            if (i - 1 >= 0):
                A[i][j - 1] = i / 10.
            if (i + 1 <= n - 1):
                A[i][j + 1] = np.e ** -(1.0 / (i + 1))

A7 = A

w, V = np.linalg.eig(A)
largest_index = 0
largest_value = 0
for i in range (n):
    if w[i] > largest_value:
        largest_value = w[i]
        largest_index = i

A8 = abs(largest_value)
A9 = V[:, largest_index]

# Problem 4

n = 48
A = np.zeros([n, n])
for i in range (n):
    for j in range (n):
        if (i == j):
            A[i][j] = 2
            if (i - 1 >= 0):
                A[i][j - 1] = -1
            if (i + 1 <= n - 1):
                A[i][j + 1] = -1     
A10 = A

rho = np.zeros([n, 1])
for i in range (n):
    rho[i][0] = 2 * (1 - np.cos((53. * np.pi) / 49.)) * np.sin((53 * np.pi * i) / 49)

A11 = rho

P = np.diag(np.diag(A))
T = A - P
M = -np.linalg.solve(P, T)
w, V = np.linalg.eig(M)

A12 = max(abs(w))
A13 = 1

def Jacobi_Gauss(P, T, rho, tol):
    k = 0
    err = tol + 1
    x_0 = np.ones((n, 1))
    X = np.zeros((n, 1))
    X[:, 0:1] = x_0
    while err >= tol:
        X = np.hstack((X, scipy.linalg.solve_triangular(P, -T @ X[:, k : (k + 1)] + rho, lower = True)))
        err = max(abs(X[:, k + 1] - X[:, k]))
        k = k + 1
    return X, X[:, k], k + 1

def error(X_sol, true_sol):
    return np.max(np.abs(true_sol - X_sol.reshape(-1, 1)))

tol = 1e-5
full_X, X_sol, k = Jacobi_Gauss(P, T, rho, tol)

A14 = X_sol
A15 = k

true_sol = np.zeros([n, 1])
for i in range (n):
    true_sol[i][0] = np.sin((53 * np.pi * i) / 49)

A16 = error(X_sol, true_sol)
print(A16)

indices = np.zeros([n, 1])
for i in range (n):
    indices[i][0] = i
plt.scatter(indices, X_sol)
plt.scatter(indices, true_sol)
plt.legend(("Jacobi Solutions", "True Solutions"))
plt.xlabel("Values")
plt.ylabel("Jacobi and True-solution Elements")
plt.title("Accuracy of the Jacobi Method by Element")
plt.show()

# Analysis: Extremely close, nearly identical. They appear to constitute the same graph.

plt.clf()
plt.scatter(np.zeros((n, 1)), full_X[:, 0])
plt.scatter(np.ones((n, 1)) * 10, full_X[:, 10])
plt.scatter(np.ones((n, 1)) * 100, full_X[:, 100])
plt.scatter(np.ones((n, 1)) * 1000, full_X[:, 1000])
plt.scatter(np.ones((n, 1)) * k, X_sol)
plt.xlabel("k-values")
plt.ylabel("Element values of Jacobi Solutions")
plt.title("Element values of Jacobi Solutions at Certain k-values")
plt.legend(("k = 0", "k = 10", "k = 100", "k = 1000", "last k"))
plt.show()

# Analysis: Gets increasingly more accurate as the k increases in value. Seems to follow an exponential decay pattern, k = 1000 already seems quite accurate.

tol = 1e-8
full_X, X_sol, k = Jacobi_Gauss(P, T, rho, tol)

A17 = X_sol
A18 = k
A19 = error(X_sol, true_sol)

P = np.tril(A)
T = A - P

tol = 1e-5
full_X, X_sol, k = Jacobi_Gauss(P, T, rho, tol)

A20 = X_sol
A21 = k
A22 = error(X_sol, true_sol)

tol = 1e-8
full_X, X_sol, k = Jacobi_Gauss(P, T, rho, tol)

A23 = X_sol
A24 = k
A25 = error(X_sol, true_sol)

plt.scatter(indices, X_sol)
plt.scatter(indices, true_sol)
plt.legend(("Gauss-Seidel Solutions", "True Solutions"))
plt.xlabel("Values")
plt.ylabel("Gauss-Seidel and True-Solution Elements")
plt.title("Accuracy of the Gauss-Seidel Method by Element")
plt.show()

plt.clf()
plt.scatter(np.zeros((n, 1)), full_X[:, 0])
plt.scatter(np.ones((n, 1)) * 5, full_X[:, 5])
plt.scatter(np.ones((n, 1)) * 10, full_X[:, 10])
plt.scatter(np.ones((n, 1)) * 20, full_X[:, 20])
plt.scatter(np.ones((n, 1)) * k, X_sol)
plt.xlabel("k-values")
plt.ylabel("Element values of Gauss-Seidel Solutions")
plt.title("Element values of Gauss-Seidel Solutions at Certain k-values")
plt.legend(("k = 0", "k = 10", "k = 20", "k = 30", "last k"))
plt.show()

# Analysis: Immediately more accurate after a couple of iterations. There is very little discernible change per iteration after the tenth.
