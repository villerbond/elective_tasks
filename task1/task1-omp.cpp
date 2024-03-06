#include <iostream>
#include <omp.h>

int main() {

    int num_threads = 5;
    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
#pragma omp critical
        {
            std::cout << "Hello from thread " << omp_get_thread_num() << std::endl;
        }
    }

    return 0;
}
