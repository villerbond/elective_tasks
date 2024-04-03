#include <iostream>
#include <omp.h>
#include <vector>

// ������ ������� f(x, y) = x^2 + y^2
double function(double x, double y) {
    return x * x + y * y;
}

void print(std::vector<std::vector<double>> matrix, const int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    const int size = 10; // ������ �����
    double h = 0.01; // ��� �����

    std::vector<std::vector<double>> A(size, std::vector<double>(size)); // �������� ������ �������� �������
    std::vector<std::vector<double>> B(size, std::vector<double>(size)); // ������ ��� �����������

    // ���������� ������� A
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i][j] = function(i * h, j * h);
        }
    }

    print(A, size);

    // ����� ������� ������ ������
    double start_time = omp_get_wtime();

    // ���������� ����������� �� ���������� x � �������������� OpenMP
#pragma omp parallel for
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - 1; j++) {
            if (i == 0) {
                B[i][j] = (A[i + 1][j] - A[i][j]) / h;
            }
            else if (i == size - 1) {
                B[i][j] = (A[i][j] - A[i - 1][j]) / h;
            }
            else {
                B[i][j] = (A[i + 1][j] - A[i - 1][j]) / (2 * h); // ����������� ���������� ����� ��� �����������
            }
        }
    }

    print(B, size);

    // ����� ������� ��������� ������
    double end_time = omp_get_wtime();

    // ����� ������� ������
    std::cout << "Size: " << size << "x" << size << std::endl;
    std::cout << "Times: " << end_time - start_time << " seconds" << std::endl;

    return 0;
}