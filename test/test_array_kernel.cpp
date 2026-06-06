#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

// NOLINTBEGIN
namespace
{
    using device_int_array = gpu_array::array<int, std::uint32_t>;
    using managed_int_array = gpu_array::managed_array<int, std::uint32_t>;

    struct kernel_record
    {
        int count = 0;
        float weight = 0.0F;
        double score = 0.0;

        friend bool operator==(const kernel_record&, const kernel_record&) = default;
    };

    void synchronize() { GPU_CHECK_ERROR(gpu_array::api::gpuDeviceSynchronize()); }

    template <typename Array>
    void expect_vector(const Array& values, const std::vector<typename Array::value_type>& expected)
    {
        EXPECT_EQ(values.template to<std::vector>(), expected);
    }

    template <typename Array>
    __global__ void write_indices_kernel(Array values, int offset)
    {
        using size_type = typename Array::size_type;
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto index = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); index < values.size();
             index += stride)
        {
            values[index] = static_cast<int>(index) + offset;
        }
    }

    template <typename Array>
    __global__ void update_with_pointer_surface_kernel(Array values, int multiplier)
    {
        using size_type = typename Array::size_type;
        const auto total = static_cast<size_type>(values.end() - values.begin());
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto index = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); index < total; index += stride)
        {
            values.data()[index] = values.begin()[index] * multiplier + static_cast<int>(total);
        }
    }

    template <typename SourceArray, typename DestinationArray>
    __global__ void copy_from_const_surface_kernel(const SourceArray source, DestinationArray destination)
    {
        using size_type = typename SourceArray::size_type;
        const auto total = static_cast<size_type>(source.end() - source.begin());
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto index = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); index < total; index += stride)
        {
            destination[index] = source.data()[index] + source.begin()[index] + static_cast<int>(total);
        }
    }

    template <typename NestedArray>
    __global__ void update_rectangular_nested_kernel(NestedArray rows, typename NestedArray::size_type columns,
                                                     int offset)
    {
        using size_type = typename NestedArray::size_type;
        const auto total = static_cast<size_type>(rows.size() * columns);
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto flat_index = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); flat_index < total;
             flat_index += stride)
        {
            const auto row_index = static_cast<size_type>(flat_index / columns);
            const auto column_index = static_cast<size_type>(flat_index % columns);
            rows[row_index][column_index] += static_cast<int>(row_index * 100 + column_index) + offset;
        }
    }

    template <typename RecordArray>
    __global__ void update_records_kernel(RecordArray records)
    {
        using size_type = typename RecordArray::size_type;
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto index = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); index < records.size();
             index += stride)
        {
            records[index].count += static_cast<int>(index) * 10;
            records[index].weight += static_cast<float>(index) * 0.5F;
            records[index].score += static_cast<double>(index) * 0.25;
        }
    }
}  // namespace

TEST(ManagedArrayKernel, WritesThroughValuePassedWrapper)
{
    constexpr auto size = std::uint32_t{37};
    auto values = managed_int_array(size, 0);

    write_indices_kernel<<<3, 8>>>(values, 5);
    synchronize();

    auto expected = std::vector<int>(size);
    for (auto index = std::size_t{0}; index < expected.size(); ++index)
    {
        expected[index] = static_cast<int>(index) + 5;
    }
    expect_vector(values, expected);
}

TEST(ArrayKernel, WritesDeviceOnlyArray)
{
    constexpr auto size = std::uint32_t{43};
    auto values = device_int_array(size, 0);

    write_indices_kernel<<<4, 7>>>(values, 11);
    synchronize();

    auto expected = std::vector<int>(size);
    for (auto index = std::size_t{0}; index < expected.size(); ++index)
    {
        expected[index] = static_cast<int>(index) + 11;
    }
    expect_vector(values, expected);
}

TEST(ArrayKernel, UsesDevicePointerAndIteratorSurface)
{
    const auto source = std::vector<int>{1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169};
    auto values = device_int_array(source);

    update_with_pointer_surface_kernel<<<2, 5>>>(values, 3);
    synchronize();

    auto expected = source;
    for (auto& value : expected)
    {
        value = value * 3 + static_cast<int>(source.size());
    }
    expect_vector(values, expected);
}

TEST(ManagedArrayKernel, ReadsConstDeviceSurface)
{
    const auto source_values = std::vector<int>{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9};
    auto source = managed_int_array(source_values);
    auto destination = managed_int_array(source_values.size(), 0);

    copy_from_const_surface_kernel<<<3, 4>>>(source, destination);
    synchronize();

    auto expected = source_values;
    for (auto& value : expected)
    {
        value = value * 2 + static_cast<int>(source_values.size());
    }
    expect_vector(destination, expected);
}

TEST(ManagedArrayKernel, UpdatesNestedRectangularArrays)
{
    constexpr auto rows = std::uint32_t{7};
    constexpr auto columns = std::uint32_t{11};
    auto source = std::vector(rows, std::vector<int>(columns, 3));
    auto values = gpu_array::managed_array(source);

    update_rectangular_nested_kernel<<<3, 9>>>(values, columns, 2);
    synchronize();

    auto expected = source;
    for (auto row_index = std::size_t{0}; row_index < expected.size(); ++row_index)
    {
        for (auto column_index = std::size_t{0}; column_index < expected[row_index].size(); ++column_index)
        {
            expected[row_index][column_index] += static_cast<int>(row_index * 100 + column_index) + 2;
        }
    }
    EXPECT_EQ(values.template to<std::vector>(), expected);
}

TEST(ManagedArrayKernel, UpdatesCustomStructFields)
{
    const auto source = std::vector<kernel_record>{{1, 1.5F, 2.5}, {2, 3.5F, 4.5}, {3, 5.5F, 6.5}, {4, 7.5F, 8.5}};
    auto records = gpu_array::managed_array<kernel_record, std::uint32_t>(source);

    update_records_kernel<<<2, 3>>>(records);
    synchronize();

    auto expected = source;
    for (auto index = std::size_t{0}; index < expected.size(); ++index)
    {
        expected[index].count += static_cast<int>(index) * 10;
        expected[index].weight += static_cast<float>(index) * 0.5F;
        expected[index].score += static_cast<double>(index) * 0.25;
    }
    expect_vector(records, expected);
}

// NOLINTEND
