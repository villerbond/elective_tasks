#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

void print(std::vector<int> vec) {
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

int main() {

    std::vector<int> numbers;

    // Заполнение массива числами
    int n = 10000000; // Размер массива
    for (int i = 0; i < n; ++i) {
        numbers.push_back(rand() % 100);
    }

    //print(numbers);

    double sum = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Подсчет суммы с использованием OpenMP
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += numbers[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "Size: " << n << std::endl;
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}
