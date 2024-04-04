#include <iostream>
#include <omp.h>
#include <vector>

// Функция для генерации матрицы случайными числами
std::vector<std::vector<int>> generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 10;
        }
    }

    return matrix;
}

// Функция умножения матриц
std::vector<std::vector<int>> matrixMultiply(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
    int n = a.size();
    int m = a[0].size();
    int p = b[0].size();

    std::vector<std::vector<int>> result(n, std::vector<int>(p, 0));

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

int main() {
    const int size = 50; // Размер матрицы
    std::vector<std::vector<int>> A = generateRandomMatrix(size, size); // Генерация случайной матрицы A
    std::vector<std::vector<int>> B = generateRandomMatrix(size, size); // Генерация случайной матрицы B

    // Засечение времени начала выполнения
    double start_time = omp_get_wtime();

    std::vector<std::vector<int>> result = matrixMultiply(A, B);

    // Засечение времени конца выполнения
    double end_time = omp_get_wtime();

    std::cout << "Size: " << size << "x" << size << std::endl;
    std::cout << "Time: " << end_time - start_time << " seconds" << std::endl;

    return 0;
}