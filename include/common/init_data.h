#pragma once

#include <random>
#include <vector>

inline void init_random(std::vector<float>& data, int seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0F, 1.0F);

    for (float& value : data)
    {
        value = distribution(generator);
    }
}
