#include "common/cuda_check.h"
#include "common/device_buffer.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <exception>

int main()
{
    try
    {
        DeviceBuffer<float> buffer(1);

        // Deliberately pass a null destination to trigger an API error.
        CUDA_CHECK(
            cudaMemcpy(nullptr, buffer.data(), sizeof(float), cudaMemcpyDeviceToHost));

        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Caught exception: %s\n", error.what());

        return 1;
    }
}
