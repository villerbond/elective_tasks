#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include <chrono>

const char* kernel_code = R"(
    __kernel void sum(__global int* numbers, __global int* result, const int n) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += numbers[i];
        }
        *result = sum;
    }
)";

int main() {
    // ������ �������
    const int arraySize = 10;

    // ������������� ������� �����
    std::vector<int> inputArray;
    for (int i = 0; i < arraySize; i++) {
        inputArray.push_back(rand() % 100);
    }

    // ��������� ��������� OpenCL
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    // ��������� ���������� OpenCL
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    // �������� ��������� OpenCL
    cl::Context context(device);

    // �������� ������� ������ OpenCL
    cl::CommandQueue queue(context, device);

    // �������� ������ ��� ������� � �������� ������
    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * arraySize, inputArray.data());
    cl::Buffer result_buffer(context, CL_MEM_WRITE_ONLY, sizeof(int));

    // �������� ��������� OpenCL
    cl::Program::Sources source(1, std::make_pair(kernel_code, strlen(kernel_code)));
    cl::Program program(context, kernel_code);

    program.build({ device });

    // �������� ���� OpenCL
    cl::Kernel kernel(program, "sum");

    // ��������� ���������� ��� ���� OpenCL
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, result_buffer);
    kernel.setArg(2, arraySize);

    // ����� �������
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // ���������� ���� OpenCL
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(arraySize), cl::NullRange, nullptr, &event);

    // ������ ����������� �� ������ ������
    int result;
    queue.enqueueReadBuffer(result_buffer, CL_TRUE, 0, sizeof(int), &result);

    event.wait();
  
    // ����� ����������
    std::cout << "Size: " << arraySize << std::endl;
    std::cout << "Sum: " << result << std::endl;

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
    std::cout << "Time: " << elapsedSeconds.count() << " seconds" << std::endl;

    return 0;
}