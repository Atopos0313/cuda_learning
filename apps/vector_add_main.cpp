#include <vector>
#include <iostream>
#include "common/init_data.h"
#include "vector_add.h"
#include "common/compare.h"

bool run_test(int N, int thread_per_block){
    std::vector<float> h_A(N);
        std::vector<float> h_B(N);
        std::vector<float> h_C(N);
        std::vector<float> cpu_C(N);
        init_random(h_A, 1);
        init_random(h_B, 2);

        vector_add(h_A.data(), h_B.data(), h_C.data(), N, thread_per_block);

        for(size_t i = 0; i < N; i++){
            cpu_C[i] = h_A[i] + h_B[i];
        }
        float eps = 1e-6f;
        return check_result(h_C, cpu_C, eps);
}

int main()
{
    std::vector<int> test_sizes = {
        0,
        1,
        1000,
        100003
    };

    std::vector<int> block_sizes = {
        64,
        128,
        256
    };

    bool all_pass = true;
    // 这个条件语句的意思是，依次取出容器中的每一个元素， 并把当前元素放进变量 block_size
    for (int block_size : block_sizes) {
        for (int N : test_sizes) {
            std::cout
                << "testing N = " << N
                << ", block size = " << block_size
                << std::endl;

            const bool ok = run_test(N, block_size);

            std::cout << (ok ? "PASS" : "FAIL") << std::endl;

            if (!ok) {
                all_pass = false;
            }
        }
    }

    return all_pass ? 0 : 1;
}
