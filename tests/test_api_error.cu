#include "common/cuda_check.h"
#include "common/device_buffer.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <exception>

int main()
{
    try {
        DeviceBuffer<float> buffer(1);

        // 故意提供空的目标地址，制造普通 API 参数错误。
        CUDA_CHECK(cudaMemcpy(
            nullptr,
            buffer.data(),
            sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        return 0;
    }
    catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "Caught exception: %s\n",
            error.what()
        );

        return 1;
    }
}