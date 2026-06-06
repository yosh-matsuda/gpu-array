#include "gpu_array.hpp"

#include <gtest/gtest.h>

// NOLINTBEGIN
namespace
{
    struct kernel_value_record
    {
        int count = 0;
        float weight = 0.0F;
        double score = 0.0;

        kernel_value_record() = default;
        __host__ __device__ kernel_value_record(int count_, float weight_, double score_)
            : count(count_), weight(weight_), score(score_)
        {
        }
    };

    void synchronize() { GPU_CHECK_ERROR(gpu_array::api::gpuDeviceSynchronize()); }

    __global__ void update_device_int_value_kernel(gpu_array::value<int> value, int offset)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            *value.get() += offset;
        }
    }

    __global__ void update_device_record_value_kernel(gpu_array::value<kernel_value_record> value)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            value.get()->count += 10;
            value.get()->weight *= 2.0F;
            value.get()->score += 0.25;
        }
    }

    __global__ void update_managed_int_value_kernel(gpu_array::managed_value<int> value, int multiplier)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            *value = *value * multiplier + 3;
        }
    }

    __global__ void update_managed_record_value_kernel(gpu_array::managed_value<kernel_value_record> value)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            value->count -= 2;
            (*value).weight += 1.5F;
            value->score *= 3.0;
        }
    }
}  // namespace

TEST(ValueKernel, MutatesDeviceOnlyValueThroughPointer)
{
    auto value = gpu_array::value<int>(17);

    update_device_int_value_kernel<<<2, 4>>>(value, 25);
    synchronize();

    EXPECT_EQ(*value, 42);
}

TEST(ValueKernel, UsesDevicePointerForRecord)
{
    auto value = gpu_array::value<kernel_value_record>(kernel_value_record{5, 1.25F, 2.5});

    update_device_record_value_kernel<<<2, 4>>>(value);
    synchronize();

    const auto actual = *value;
    EXPECT_EQ(actual.count, 15);
    EXPECT_FLOAT_EQ(actual.weight, 2.5F);
    EXPECT_DOUBLE_EQ(actual.score, 2.75);
}

TEST(ManagedValueKernel, MutatesUnifiedValue)
{
    auto value = gpu_array::managed_value<int>(13);

    update_managed_int_value_kernel<<<2, 4>>>(value, 3);
    synchronize();

    EXPECT_EQ(*value, 42);
}

TEST(ManagedValueKernel, UsesDeviceArrowForRecord)
{
    auto value = gpu_array::managed_value<kernel_value_record>(kernel_value_record{9, 2.5F, 1.25});

    update_managed_record_value_kernel<<<2, 4>>>(value);
    synchronize();

    EXPECT_EQ(value->count, 7);
    EXPECT_FLOAT_EQ(value->weight, 4.0F);
    EXPECT_DOUBLE_EQ(value->score, 3.75);
}

// NOLINTEND
