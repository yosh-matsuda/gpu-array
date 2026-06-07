#include "gpu_array.hpp"

#include "constrained_kernel_launch.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
#include <ranges>
#include <vector>

// NOLINTBEGIN
namespace
{
    using int_array = gpu_array::managed_array<int, std::uint32_t>;

    void synchronize() { GPU_CHECK_ERROR(gpu_array::api::gpuDeviceSynchronize()); }

    template <typename Range>
    __global__ void update_grid_thread_stride_kernel(Range values)
    {
        for (auto& value : values | gpu_array::views::grid_thread_stride)
        {
            value = value * 2 + 1;
        }
    }

    template <typename Range>
    __global__ void update_nested_stride_kernel(Range values, int fill_value)
    {
        for (auto& row : values | gpu_array::views::grid_block_stride)
        {
            for (auto& value : row | gpu_array::views::block_thread_stride)
            {
                value = fill_value;
            }
        }
    }

#if !defined(ENABLE_HIP)
    template <typename Range>
    __global__ void update_nested_stride_alias_kernel(Range values, int fill_value)
    {
        for (auto& row : gpu_array::grid_block_stride_view(values))
        {
            for (auto& value : gpu_array::block_thread_stride_view(row))
            {
                value = fill_value;
            }
        }
    }
#endif

    template <typename Range, typename Output>
    __global__ void enumerate_kernel(Range values, Output output)
    {
        for (auto&& [index, value] : values | gpu_array::views::enumerate)
        {
            output[index] = value * static_cast<int>(index + 1);
        }
    }

    template <typename Range>
    __global__ void enumerate_stride_kernel(Range values)
    {
        for (auto&& [row_index, row] : values | gpu_array::views::enumerate | gpu_array::views::grid_block_stride)
        {
            for (auto&& [column_index, value] :
                 row | gpu_array::views::enumerate | gpu_array::views::block_thread_stride)
            {
                value = static_cast<int>(row_index * 100 + column_index);
            }
        }
    }

    template <typename Lhs, typename Rhs>
    __global__ void zip_kernel(Lhs lhs, Rhs rhs)
    {
        for (auto&& [left, right] : gpu_array::views::zip(lhs, rhs))
        {
            left += right;
        }
    }

    template <typename Range>
    __global__ void initialize_nested_kernel(Range values, int coefficient)
    {
        for (auto&& [row_index, row] : values | gpu_array::views::enumerate | gpu_array::views::grid_block_stride)
        {
            for (auto&& [column_index, value] :
                 row | gpu_array::views::enumerate | gpu_array::views::block_thread_stride)
            {
                value = static_cast<int>((row_index * row.size() + column_index) * coefficient);
            }
        }
    }

    template <typename Lhs, typename Rhs>
    __global__ void zip_stride_kernel(Lhs lhs, Rhs rhs)
    {
        for (auto&& [left_row, right_row] : gpu_array::views::zip(lhs, rhs) | gpu_array::views::grid_block_stride)
        {
            for (auto&& [left, right] :
                 gpu_array::views::zip(left_row, right_row) | gpu_array::views::block_thread_stride)
            {
                left += right;
            }
        }
    }

    template <typename Lhs, typename Rhs>
    __global__ void zip_enumerate_stride_kernel(Lhs lhs, Rhs rhs)
    {
        for (auto&& [index, zipped] :
             gpu_array::views::zip(lhs, rhs) | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride)
        {
            auto&& [left, right] = zipped;
            left = left * 100 + right * static_cast<int>(index + 1);
        }
    }

    template <typename Lhs, typename Rhs>
    __global__ void enumerate_zip_stride_kernel(Lhs lhs, Rhs rhs)
    {
        for (auto&& [enumerated, right] :
             gpu_array::views::zip(lhs | gpu_array::views::enumerate, rhs) | gpu_array::views::grid_thread_stride)
        {
            auto&& [index, left] = enumerated;
            left = left * 100 + right * static_cast<int>(index + 1);
        }
    }
}  // namespace

TEST(ViewKernel, GridThreadStride)
{
    const auto source = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto values = int_array(source);

    gpu_array_test::launch_input_range<update_grid_thread_stride_kernel<decltype(values)>>({3}, {5}, values);
    synchronize();

    auto expected = source;
    for (auto& value : expected)
    {
        value = value * 2 + 1;
    }
    EXPECT_EQ(values.template to<std::vector>(), expected);
}

TEST(ViewKernel, NestedStride)
{
    auto values = gpu_array::managed_array(std::vector(7, std::vector<int>(11, 0)));

    gpu_array_test::launch_nested_input_range<update_nested_stride_kernel<decltype(values)>>({3}, {5}, values, 17);
    synchronize();

    for (const auto& row : values)
    {
        for (const auto& value : row)
        {
            EXPECT_EQ(value, 17);
        }
    }
}

#if !defined(ENABLE_HIP)
TEST(ViewKernel, StrideAliases)
{
    auto values = gpu_array::managed_array(std::vector(5, std::vector<int>(9, 0)));

    gpu_array_test::launch_nested_input_range<update_nested_stride_alias_kernel<decltype(values)>>({2}, {4}, values,
                                                                                                    23);
    synchronize();

    for (const auto& row : values)
    {
        for (const auto& value : row)
        {
            EXPECT_EQ(value, 23);
        }
    }
}
#endif

TEST(ViewKernel, Enumerate)
{
    const auto source = std::vector<int>{16, 6, 14, 17, 5};
    auto values = int_array(source);
    auto output = int_array(source.size());

    gpu_array_test::launch_input_range_pair<enumerate_kernel<decltype(values), decltype(output)>>({1}, {1}, values,
                                                                                                  output);
    synchronize();

    auto expected = std::vector<int>{};
    expected.reserve(source.size());
    for (auto index = std::size_t{0}; index < source.size(); ++index)
    {
        expected.push_back(source[index] * static_cast<int>(index + 1));
    }
    EXPECT_EQ(output.template to<std::vector>(), expected);
}

TEST(ViewKernel, EnumerateStride)
{
    auto values = gpu_array::managed_array(std::vector(7, std::vector<int>(11, 0)));

    gpu_array_test::launch_nested_input_range<enumerate_stride_kernel<decltype(values)>>({3}, {5}, values);
    synchronize();

    for (auto row_index = std::size_t{0}; row_index < values.size(); ++row_index)
    {
        for (auto column_index = std::size_t{0}; column_index < values[row_index].size(); ++column_index)
        {
            EXPECT_EQ(values[row_index][column_index], static_cast<int>(row_index * 100 + column_index));
        }
    }
}

TEST(ViewKernel, Zip)
{
    auto lhs = int_array(std::vector<int>{19, 70, 86, 69, 42});
    auto rhs = int_array(std::vector<int>{16, 6, 14});

    gpu_array_test::launch_input_range_pair<zip_kernel<decltype(lhs), decltype(rhs)>>({1}, {1}, lhs, rhs);
    synchronize();

    EXPECT_EQ(lhs.template to<std::vector>(), (std::vector<int>{35, 76, 100, 69, 42}));
}

TEST(ViewKernel, ZipStride)
{
    auto lhs = gpu_array::managed_array(std::vector(7, std::vector<int>(11, 0)));
    auto rhs = gpu_array::managed_array(std::vector(7, std::vector<int>(11, 0)));

    gpu_array_test::launch_nested_input_range<initialize_nested_kernel<decltype(lhs)>>({3}, {5}, lhs, 1);
    gpu_array_test::launch_nested_input_range<initialize_nested_kernel<decltype(rhs)>>({3}, {5}, rhs, 1000);
    synchronize();

    gpu_array_test::launch_nested_input_range_pair<zip_stride_kernel<decltype(lhs), decltype(rhs)>>({3}, {5}, lhs,
                                                                                                    rhs);
    synchronize();

    for (auto row_index = std::size_t{0}; row_index < lhs.size(); ++row_index)
    {
        for (auto column_index = std::size_t{0}; column_index < lhs[row_index].size(); ++column_index)
        {
            EXPECT_EQ(lhs[row_index][column_index], static_cast<int>((row_index * 11 + column_index) * 1001));
        }
    }
}

TEST(ViewKernel, ZipEnumerateStride)
{
    const auto lhs_source = std::vector<int>{19, 70, 86, 69, 42};
    const auto rhs_source = std::vector<int>{16, 6, 14};
    auto lhs = int_array(lhs_source);
    auto rhs = int_array(rhs_source);

    gpu_array_test::launch_input_range_pair<zip_enumerate_stride_kernel<decltype(lhs), decltype(rhs)>>({1}, {2}, lhs,
                                                                                                       rhs);
    synchronize();

    EXPECT_EQ(lhs.template to<std::vector>(), (std::vector<int>{1916, 7012, 8642, 69, 42}));
}

TEST(ViewKernel, EnumerateZipStride)
{
    const auto lhs_source = std::vector<int>{19, 70, 86, 69, 42};
    const auto rhs_source = std::vector<int>{16, 6, 14};
    auto lhs = int_array(lhs_source);
    auto rhs = int_array(rhs_source);

    gpu_array_test::launch_input_range_pair<enumerate_zip_stride_kernel<decltype(lhs), decltype(rhs)>>({1}, {2}, lhs,
                                                                                                       rhs);
    synchronize();

    EXPECT_EQ(lhs.template to<std::vector>(), (std::vector<int>{1916, 7012, 8642, 69, 42}));
}

// NOLINTEND
