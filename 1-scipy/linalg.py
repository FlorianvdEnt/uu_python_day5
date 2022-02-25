import numpy as np
from scipy import linalg, optimize

A = np.array(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
        , dtype=np.float64)

b = np.array([1, 2, 3]) 

solution = linalg.lstsq(A, b)

print('\t1c. Solve the linear system of equations A x = b')
print(f'solution: {solution}')

print('\n\t1d. Check that your solution is correct by plugging it into the equation')
print(f'A * solution gives: {np.matmul(A, solution[0])}')


print('\n\t1e. Repeat steps a-d using a random 3x3 matrix B (instead of the vector b)')
B = np.random.randint(10, size=[3, 3])
print(f'random array B:\n {B}')
solution = linalg.lstsq(A, B)
print(f'solution:\n {solution[0]}')
print(f'np.matmul(A, solution[0]) result:\n {np.matmul(A, solution[0])}')


print('\n\t1f. Solve the eigenvalue problem for the matrix A and print the eigenvalues and eigenvectors')

print(linalg.eig(A))

print('\n\t1g. Calculate the inverse, determinant of A')
print(f'inverse: {linalg.inv(B)}')
print(f'determinant: {linalg.det(B)}')

print('\n\t1h. Calculate the norm of A with different orders')
print(f'the fck are orders in this context?')
print(f'{linalg.norm(A)}')


