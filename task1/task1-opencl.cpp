#include <CL/cl.hpp>
#include <iostream>

// ��� ��������� �� OpenCL
const char* kernelSource = R"(
__kernel void helloWorld(__global int* id) {
    int threadId = get_global_id(0);
    printf("Hello from thread %d\n", threadId);
}
)";

int main() {
    // �������� ������ ��������� OpenCL (� ������ ������ ���������� AMD)
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    // ������� �������� � ������� ������
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // ������� ����� ������
    int numThreads = 10;
    cl::Buffer buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * numThreads);

    // ����������� � ��������� ��������� �� OpenCL
    cl::Program::Sources source(1, std::make_pair(kernelSource, strlen(kernelSource)));
    cl::Program program(context, source);
    program.build({ device });
    cl::Kernel kernel(program, "helloWorld");

    kernel.setArg(0, buffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numThreads), cl::NullRange);
    queue.finish();

    // ��������� ������ ������� �� CPU � ������� ���������
    int* result = new int[numThreads];
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(int) * numThreads, result);
    for (int i = 0; i < numThreads; i++) {
        std::cout << "Result from thread " << i << ": " << result[i] << std::endl;
    }

    delete[] result;
    return 0;
}