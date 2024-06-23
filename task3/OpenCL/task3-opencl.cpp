#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include <chrono>

const char* kernel_code = R"(
    __kernel void compute_derivative_x(__global const float* A, __global float* B, const int width, const int height) {
        int i = get_global_id(0);
        int j = get_global_id(1);

        if (i > 0 && i < width - 1) {
            B[j * width + i] = (A[j * width + (i + 1)] - A[j * width + (i - 1)]) / 2.0f;
        }
        else {
            B[j * width + i] = 0.0f; // �������
        }
    }
)";

// ������ ������� f(x, y) = x^2 + y^2
double function(double x, double y) {
    return x * x + y * y;
}

int main() {

    // ������� �������
    int width = 10000;
    int height = 10000;

    // ������������� ������
    std::vector<float> A(width * height);
    std::vector<float> B(width * height);

    // ���������� ������� A ���������� ������� f(i, j)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            A[j * width + i] = function(i, j);
        }
    }

    // ��������� �������� � ���������
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    // �������� ��������� � ������� ������
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // �������� ��������� � ����
    cl::Program::Sources source(1, std::make_pair(kernel_code, strlen(kernel_code)));
    cl::Program program(context, kernel_code);
    program.build({ device });
    cl::Kernel kernel(program, "compute_derivative_x");

    // �������� �������
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), A.data());
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(float) * B.size());

    // ��������� ���������� ����
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, width);
    kernel.setArg(3, height);

    // ������ ����
    cl::NDRange global(width, height);
    auto start = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    queue.finish();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // ����������� ���������� �������
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(float) * B.size(), B.data());

    std::cout << "Time: " << diff.count() << " seconds\n";

    // ����� ����� ���������� ��� ��������
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            std::cout << B[j * width + i] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
