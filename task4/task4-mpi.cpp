#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

// ������� ��� ������������� ������� ���������� ����������
void fill_matrix(std::vector<double>& matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int N = 500; // ������ ������� �� ���������
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    std::vector<double> A, B, C;
    if (world_rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);
        fill_matrix(A, N);
        fill_matrix(B, N);
    }

    int rows_per_proc = N / world_size;
    std::vector<double> local_A(rows_per_proc * N);
    std::vector<double> local_C(rows_per_proc * N);

    auto start = std::chrono::high_resolution_clock::now();

    // ���������� ������� A ����� ����������
    MPI_Scatter(A.data(), rows_per_proc * N, MPI_DOUBLE, local_A.data(), rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // �������� ���� ������� B ������� ��������
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ���������� ������������ ��������� ������
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < N; ++j) {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < N; ++k) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    // ���� ��������� ������ ���������� � �������� 0
    MPI_Gather(local_C.data(), rows_per_proc * N, MPI_DOUBLE, C.data(), rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;


    if (world_rank == 0) {
        std::cout << "Time: " << diff.count() << " seconds\n";
        // ����� ����� ���������� ��� ��������
        //for (int i = 0; i < 10; ++i) {
        //    for (int j = 0; j < 10; ++j) {
        //        std::cout << C[i * N + j] << " ";
        //    }
        //    std::cout << "\n";
        //}
    }

    MPI_Finalize();
    return 0;
}