#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include <chrono>

const char* kernel_code = R"(
    __kernel void matrix_mult(__global const float* A, __global const float* B, __global float* C, const int N) {
        int row = get_global_id(0);
        int col = get_global_id(1);

        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
)";

int main() {
    // Размер матриц (N x N)
    int N = 1000;

    // Инициализация данных
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N);

    // Заполнение матриц A и B случайными значениями
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Получение платформ и устройств
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    // Создание контекста и очереди команд
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Создание программы и ядра
    cl::Program::Sources source(1, std::make_pair(kernel_code, strlen(kernel_code)));
    cl::Program program(context, kernel_code);
    program.build({ device });
    cl::Kernel kernel(program, "matrix_mult");

    // Создание буферов
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), A.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * B.size(), B.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * C.size());

    // Установка аргументов ядра
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, N);

    // Запуск ядра
    cl::NDRange global(N, N);
    auto start = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    queue.finish();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Копирование результата обратно
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * C.size(), C.data());

    std::cout << "Time: " << diff.count() << " seconds\n";

    // Вывод части результата для проверки

    //for (int i = 0; i < 10; ++i) {
    //    for (int j = 0; j < 10; ++j) {
    //        std::cout << A[i * N + j] << " ";
    //    }
    //    std::cout << "\n";
    //}

    //for (int i = 0; i < 10; ++i) {
    //    for (int j = 0; j < 10; ++j) {
    //        std::cout << B[i * N + j] << " ";
    //    }
    //    std::cout << "\n";
    //}

    //for (int i = 0; i < 10; ++i) {
    //    for (int j = 0; j < 10; ++j) {
    //        std::cout << C[i * N + j] << " ";
    //    }
    //    std::cout << "\n";
    //}

    return 0;
}
