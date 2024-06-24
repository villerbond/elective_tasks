#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

// Пример функции f(x, y) = x^2 + y^2
double function(double x, double y) {
    return x * x + y * y;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Размеры массива
    int width = 10000;
    int height = 10000;
    if (argc > 2) {
        width = std::atoi(argv[1]);
        height = std::atoi(argv[2]);
    }

    std::vector<double> A;
    if (world_rank == 0) {
        A.resize(width * height);
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                A[j * width + i] = function(i, j);
            }
        }
    }

    int local_height = height / world_size;
    std::vector<double> local_A(local_height * width);

    auto start = std::chrono::high_resolution_clock::now();

    // Разделение данных между процессами
    MPI_Scatter(A.data(), local_height * width, MPI_DOUBLE, local_A.data(), local_height * width, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычисление производной по x
    std::vector<double> local_B(local_height * width);
    for (int j = 0; j < local_height; ++j) {
        for (int i = 1; i < width - 1; ++i) {
            local_B[j * width + i] = (local_A[j * width + (i + 1)] - local_A[j * width + (i - 1)]) / 2.0;
        }
    }

    // Собираем данные обратно
    std::vector<double> B;
    if (world_rank == 0) {
        B.resize(width * height);
    }
    MPI_Gather(local_B.data(), local_height * width, MPI_DOUBLE, B.data(), local_height * width, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (world_rank == 0) {
        std::cout << "Time: " << diff.count() << " seconds\n";
        // Вывод части результата для проверки
        for (int j = 0; j < 10; ++j) {
            for (int i = 0; i < 10; ++i) {
                std::cout << B[j * width + i] << " ";
            }
            std::cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
