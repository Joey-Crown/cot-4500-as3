import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

# Question 1
f = lambda t, y : t - (y**2)
a = 0
b = 2
alpha = 1
n = 10

def euler_mehthod(a, b, n, alpha):
    h = (b - a) / n
    t = a
    w = alpha
    for i in range(n):
        w = w + (h * f(t, w))
        t = t + h
    return w

ans_q1 = euler_mehthod(a, b, n, alpha)
print("%.5f" % ans_q1, end="\n\n")

# Question 2
def runge_kutta(a, b, n, alpha):
    h = (b - a) / n
    t = a
    w = alpha
    for i in range(n):
        k1 = h * f(t, w)
        k2 = h * f(t + h / 2, w + k1 / 2)
        k3 = h * f(t + h / 2, w + k2 / 2)
        k4 = h * f(t + h, w + k3)

        w = w + (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + h
    return w


ans_q2 = runge_kutta(a, b, n, alpha)
print("%.5f" % ans_q2, end="\n\n")

# Question 3
augmented_matrix = np.array([[2,-1,1,6],
                             [1,3,1,0],
                             [-1,5,4,-3]], dtype=float)
n = len(augmented_matrix)

def gauss_jordan(Ab, n):
    for i in range(n):
        row = i
        for j in range(i+1, n):
            if abs(Ab[j,i]) > abs(Ab[row,i]):
                row = j
        # swap rows
        Ab[[i, row]] = Ab[[row, i]]

        # divide row by pivot element
        Ab[i] = Ab[i] / Ab[i, i]

        # elimnate element below pivot
        for j in range(i+1, n):
            Ab[j] -= Ab[j, i] * Ab[i]

    for i in range(n - 1, -1, -1):
        for j in  range(i - 1, -1, -1):
            Ab[j] -= Ab[j, i] * Ab[i]
    
    return Ab[:,n]

ans_q3 = gauss_jordan(augmented_matrix, n)
print(ans_q3, end="\n\n")


# Question 4
matrix = np.array([[1,1,0,3],
                   [2,1,-1,1],
                   [3,-1,-1,2],
                   [-1,2,3,-1]], dtype=float)

def LU_factorization(A):
    n = len(A)
    L = np.zeros((n, n), float)
    U = np.zeros((n, n), float)
    np.fill_diagonal(L, 1)
    
    for i in range(n - 1):
        U[i] = A[i]
        L[i+1:,i] = A[i+1:,i] / U[i,i]
        # divide row by pivot element
        A[i] = A[i] / A[i, i]

        # elimnate element below pivot
        for j in range(i+1, n):
            A[j] -= A[j, i] * A[i]

    # add last row to U
    U[n - 1] = A[n - 1]
    return L, U

ans_q4_b, ans_q4_c = LU_factorization(matrix)
# get determinant by multiply the diagonal of U
ans_q4_a = np.prod(np.diagonal(ans_q4_c))
print("%.5f" % ans_q4_a, end="\n\n")
print(ans_q4_b, end="\n\n")
print(ans_q4_c, end="\n\n")

# Question 5
matrix = np.array([[9,0,5,2,1],
                   [3,9,1,2,1],
                   [0,1,7,2,3],
                   [4,2,3,12,2],
                   [3,2,4,0,8]], dtype=float)

def diagonal_dominance(A):
    n = len(A)
    for i in range(n):
        sum = 0
        for j in range(n):
            if j != i:
                sum += abs(A[i,j])
        if (sum > abs(A[i,i])):
            return False
    return True

ans_q5 = diagonal_dominance(matrix)
print(ans_q5, end="\n\n")

# Question 6
matrix = np.array([[2,2,1],
                   [2,3,0],
                   [1,0,2]])

def positive_definite(A):
    n = len(A)
    L = np.zeros((n,n), float)
    np.fill_diagonal(L, 1)
    D = np.zeros(3, float)
    v = np.zeros(3, float)
    for i in range(n):
        for j in range(i):
            v[j] = L[i,j] * D[j]
        sum = 0

        for j in range(i):
            sum += L[i,j] * v[j]

        D[i] = A[i,i] - sum

        for j in range(i + 1, n):
            sum = 0
            for k in range(i):
                sum += L[j,k] * v[k]
            L[j,i] = (A[j,i] - sum) / D[i]
    
    for i in range(n):
        if D[i] < 0:
            return False
    return True

ans_q6 = positive_definite(matrix)
print(ans_q6)
