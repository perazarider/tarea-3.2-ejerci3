import numpy as np

def gauss_jordan_pivot_determinante(A, b):
    """
    Resuelve un sistema de ecuaciones Ax = b mediante el método de Gauss-Jordan con pivoteo parcial
    e imprime el determinante de A para verificar si el sistema tiene solución única.
    """
    n = len(A)
    
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)
    
    det_A = np.linalg.det(A)
    
    if np.isclose(det_A, 0):
        mensaje = f"Determinante de A: {det_A:.5f}. El sistema es indeterminado o no tiene solución única."
        print(mensaje)
        return None
    
    mensaje = f"Determinante de A: {det_A:.5f}. El sistema tiene solución única."
    print(mensaje)
    
    for i in range(n):
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        Ab[i] = Ab[i] / Ab[i, i]

        for j in range(n):
            if i != j:
                Ab[j] -= Ab[j, i] * Ab[i]

    x = Ab[:, -1]
    return x

A_test = np.array([[2, -3, 4, -1, 5, -1, 2, -1, 3, -2],
                   [-3, 2, 5, -1, 4, 2, -3, 1, -2, 5],
                   [4, -1, 3, 2, -3, 1, -2, 5, -4, 1],
                   [-1, 5, -2, 3, 4, -1, 2, -3, 1, -5],
                   [3, -2, 5, -1, 4, 2, -3, 1, -5, 2],
                   [-2, 4, -3, 1, 5, -1, 2, -4, 3, -1],
                   [5, -1, 2, -3, 4, 1, -2, 3, -1, 4],
                   [1, -3, 4, -2, 5, -1, 2, -1, 4, -3],
                   [2, 3, -1, 4, -2, 5, -3, 1, -2, 1],
                   [-3, 2, 4, -1, 3, -2, 5, -1, 1, -4]], dtype=float)

b_test = np.array([11, -10, 8, -6, 7, -3, 9, -5, 6, -8], dtype=float)

solucion_test = gauss_jordan_pivot_determinante(A_test, b_test)

if solucion_test is not None:
    print("Solución del sistema:", solucion_test)
