#include <iostream>
#include <vector>
#include <ctime>
#include "mpi.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10; // Размер массива
    std::vector<int> numbers; // Инициализируем массив чисел

    for (int i = 0; i < n; ++i) {
        numbers.push_back(i);
    }

    for (int i = 0; i < n; ++i) {
        std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;

    double start_time = MPI_Wtime();

    int local_sum = 0;
    for (int i = 0; i < n; ++i) {
        local_sum += numbers[i];
    }

    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end_time = MPI_Wtime();
        
        std::cout << "Size: " << n << std::endl;
        std::cout << "Sum: " << global_sum << std::endl;
        std::cout << "Time: " << end_time - start_time << " seconds" << std::endl;
    }

    MPI_Finalize();

    return 0;
}