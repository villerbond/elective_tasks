#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>

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

void print(std::vector<std::vector<int>> matrix, const int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


int main(int argc, char** argv) {

    // Статус отправки/получения
    MPI_Status status;

    MPI_Init(&argc, &argv);

    int rank, size=10;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int el_per_process, ind, i;

    const int N = 10; // Размер матрицы NxN
    std::vector<std::vector<int>> A = generateRandomMatrix(N, N); // Генерация случайной матрицы A
    std::vector<std::vector<int>> B = generateRandomMatrix(N, N); // Генерация случайной матрицы B

    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    if (rank == 0) {

        auto start_time = std::chrono::high_resolution_clock::now();

        el_per_process = N / size;
        ind = el_per_process;
        int tmp;

        // Раздача задач
        if (size > 1) {
            for (i = 1; i < size - 1; ++i) {
                MPI_Send(&ind, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&el_per_process, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&A[ind][0], el_per_process * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
                MPI_Send(&B, N * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
                ind += el_per_process;
                tmp = i;
            }
            int els_left = N - ind;
            MPI_Send(&ind, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&els_left, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&A[ind][0], els_left * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            MPI_Send(&B, N * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }


        // Собираем результаты воедино

        for (i = 1; i < size; i++) {
            MPI_Recv(&ind, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&el_per_process, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[ind][0], el_per_process * N, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        std::cout << "Size: " << N << "x" << N << std::endl;
        std::cout << "Time: " << duration.count() << " seconds" << std::endl;
    } else {
        int src = 0;
        MPI_Recv(&ind, 1, MPI_INT, src, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&el_per_process, 1, MPI_INT, src, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&A, el_per_process * N, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&B, N * N, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, &status);

        // Умножение матриц
        for (i = 0; i < el_per_process; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        MPI_Send(&ind, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&el_per_process, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&C, el_per_process * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
