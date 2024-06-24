#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int N = 10000000;  // Размер массива по умолчанию
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    std::vector<int> data;
    if (world_rank == 0) {
        data.resize(N);
        for (int i = 0; i < N; ++i) {
            data[i] = i;
        }
    }

    int local_N = N / world_size;
    std::vector<int> local_data(local_N);

    auto start = std::chrono::high_resolution_clock::now();

    // Разделение данных между процессами
    MPI_Scatter(data.data(), local_N, MPI_INT, local_data.data(), local_N, MPI_INT, 0, MPI_COMM_WORLD);

    // Локальное суммирование
    int local_sum = std::accumulate(local_data.begin(), local_data.end(), 0);

    // Сбор локальных сумм и суммирование
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (world_rank == 0) {
        std::cout << "Sum: " << global_sum << "\n";
        std::cout << "Time: " << diff.count() << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}