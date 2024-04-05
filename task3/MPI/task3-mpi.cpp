#include <iostream>
#include <mpi.h>
#include <vector>
#include <chrono>

// Пример функции f(x, y) = x^2 + y^2
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

int main(int argc, char** argv) {
    // Статус отправки/получения
    MPI_Status status;
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int size_of_matrix = 10000; // размер сетки
    double h = 0.01; // шаг сетки

    std::vector<std::vector<double>> A(size_of_matrix, std::vector<double>(size_of_matrix)); // исходный массив значений функции
    std::vector<std::vector<double>> B(size_of_matrix, std::vector<double>(size_of_matrix)); // массив для производной

    // Заполнение массива A
    for (int i = 0; i < size_of_matrix; i++) {
        for (int j = 0; j < size_of_matrix; j++) {
            A[i][j] = function(i * h, j * h);
        }
    }

    //print(A, size_of_matrix);

    // Главный процесс
    if (rank == 0) {
        // Заполнение массива A
        for (int i = 0; i < size_of_matrix; i++) {
            for (int j = 0; j < size_of_matrix; j++) {
                A[i][j] = function(i * h, j * h);
            }
        }
        int i, index, n_elements_recieved;
        int elements_per_process = size_of_matrix / size;

        // Замер времени начала выполнения
        auto start_time = std::chrono::high_resolution_clock::now();

        // Раздача задач
        if (size > 1) {
            for (i = 1; i < size - 1; i++) {
                int index = i * elements_per_process;

                MPI_Send(&elements_per_process, 1, MPI_INT, i, 0,
                    MPI_COMM_WORLD);
                MPI_Send(&A[index][0], elements_per_process * size_of_matrix,
                    MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&index, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            index = i * elements_per_process;
            int elements_left = size_of_matrix - index;
            MPI_Send(&elements_left, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&A[index][0], elements_left * size_of_matrix, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&index, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // Считаем самую первую часть производной

        for (int i = 0; i < elements_per_process; i++) {
            for (int j = 0; j < size_of_matrix; j++) {
                if (i == 0) {
                    B[i][j] = (A[i + 1][j] - A[i][j]) / h;
                }
                else if (i == size_of_matrix - 1) {
                    B[i][j] = (A[i][j] - A[i - 1][j]) / h;
                }
                else {
                    B[i][j] = (A[i + 1][j] - A[i - 1][j]) / (2 * h); // Центральная разностная схема для производной
                }
            }
        }

        // Получаем результаты производных от других процессам
        for (int i = 1; i < rank; i++) {
            MPI_Recv(&index, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&n_elements_recieved, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&B[index][0], n_elements_recieved * size_of_matrix, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
        }

        // Замер окончания работы
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Size: " << size_of_matrix << "x" << size_of_matrix;
        std::cout << " Time: " << (elapsed.count()) << " seconds" << std::endl;

    }
    // Работа неосновных процессов
    else {
        int n_elements_recieved, index;
        MPI_Recv(&n_elements_recieved, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
            &status);

        MPI_Recv(&A, n_elements_recieved * size_of_matrix, MPI_DOUBLE, 0, 0,
            MPI_COMM_WORLD, &status);
        MPI_Recv(&index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // Вычисление производной для полученной части массива

        for (int i = 0; i < n_elements_recieved; i++) {
            for (int j = 0; j < size_of_matrix; j++) {
                if (i == 0) {
                    B[i][j] = (A[i + 1][j] - A[i][j]) / h;
                }
                else if (i == size_of_matrix - 1) {
                    B[i][j] = (A[i][j] - A[i - 1][j]) / h;
                }
                else {
                    B[i][j] = (A[i + 1][j] - A[i - 1][j]) / (2 * h); // Центральная разностная схема для производной
                }
            }
        }

        // Отправка вычисленных производных в главный процесс

        MPI_Send(&index, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&n_elements_recieved, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&B, n_elements_recieved * size_of_matrix, MPI_DOUBLE, 0, 1,
            MPI_COMM_WORLD);
    }

    //print(B, size_of_matrix);

    return 0;
}
