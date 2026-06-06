#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
#include <vector>

// NOLINTBEGIN
namespace
{
    using namespace gpu_array;

    void synchronize() { GPU_CHECK_ERROR(api::gpuDeviceSynchronize()); }

    template <std::ranges::input_range Range>
    __global__ void readme_increment_kernel(Range array)
    {
        for (auto& value : array | views::grid_thread_stride)
        {
            value += 1;
        }
    }

    template <std::ranges::input_range Range>
    requires std::ranges::input_range<std::ranges::range_value_t<Range>>
    __global__ void readme_nested_kernel(Range array)
    {
        for (auto& row : array | views::grid_block_stride)
        {
            for (auto& value : row | views::block_thread_stride)
            {
                value *= 2;
            }
        }
    }

    __global__ void readme_kernel_with_index(managed_array<int> array)
    {
        for (auto&& [index, value] : array | views::enumerate | views::grid_thread_stride)
        {
            value += static_cast<int>(index);
        }
    }

    __global__ void readme_add_kernel(managed_array<int> lhs, managed_array<int> rhs)
    {
        for (auto&& [left, right] : views::zip(lhs, rhs) | views::grid_thread_stride)
        {
            left += right;
        }
    }

    template <typename... Ts>
    requires (sizeof...(Ts) == 3)
    struct CustomTuple : public tuple<Ts...>
    {
        using tuple<Ts...>::tuple;
        using tuple<Ts...>::operator=;

        __host__ __device__ auto& get_a() { return get<0>(*this); }
        __host__ __device__ auto& get_b() { return get<1>(*this); }
        __host__ __device__ auto& get_c() { return get<2>(*this); }
    };

    using Struct = CustomTuple<int, float, double>;

    template <typename Range>
    __global__ void readme_soa_kernel(Range array)
    {
        for (auto&& value : array | views::grid_thread_stride)
        {
            value.get_a() *= 2;
            value.get_b() *= 2.0F;
            value.get_c() *= 2.0;
        }
    }

    __global__ void readme_raw_pointer_kernel(int* data, std::uint32_t size)
    {
        const auto block = cooperative_groups::this_thread_block();
        for (auto index = static_cast<std::uint32_t>(block.thread_rank()); index < size;
             index += static_cast<std::uint32_t>(block.size()))
        {
            data[index] = 1;
        }
    }

    __global__ void readme_array_kernel(managed_array<int, std::uint32_t> array)
    {
        const auto block = cooperative_groups::this_thread_block();
        for (auto index = static_cast<std::uint32_t>(block.thread_rank()); index < array.size();
             index += static_cast<std::uint32_t>(block.size()))
        {
            array[index] = 1;
        }
    }

    __global__ void readme_array_view_kernel(managed_array<int, std::uint32_t> array)
    {
        for (auto& value : array | views::block_thread_stride)
        {
            value = 1;
        }
    }

    template <typename Record>
    void expect_records_eq(const std::vector<Record>& actual, const std::vector<Record>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (auto index = std::size_t{0}; index < actual.size(); ++index)
        {
            EXPECT_EQ(get<0>(actual[index]), get<0>(expected[index]));
            EXPECT_FLOAT_EQ(get<1>(actual[index]), get<1>(expected[index]));
            EXPECT_DOUBLE_EQ(get<2>(actual[index]), get<2>(expected[index]));
        }
    }
}  // namespace

TEST(ReadmeExamples, DeviceMemoryManagement)
{
    auto array = managed_array<int>(8);

    readme_increment_kernel<<<1, 4>>>(array);
    synchronize();

    EXPECT_EQ(array.template to<std::vector>(), (std::vector<int>{1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(ReadmeExamples, HostDeviceConversions)
{
    auto values = std::vector<int>(8);
    for (auto index = 0; auto& value : values)
    {
        value = index++;
    }

    auto array = managed_array(values);

    readme_increment_kernel<<<1, 4>>>(array);
    synchronize();

    values = array.to<std::vector>();
    EXPECT_EQ(values, (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(ReadmeExamples, GridStrideAdapters)
{
    auto nested = std::vector(5, std::vector<int>(7, 1));
    auto array = managed_array(nested);

    readme_nested_kernel<<<3, 4>>>(array);
    synchronize();

    EXPECT_EQ(array.template to<std::vector>(), (std::vector(5, std::vector<int>(7, 2))));
}

TEST(ReadmeExamples, EnumerateAndZip)
{
    auto values = managed_array<int>(std::vector<int>{1, 1, 1, 1, 1});
    readme_kernel_with_index<<<2, 3>>>(values);
    synchronize();

    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{1, 2, 3, 4, 5}));

    auto rhs = managed_array<int>(std::vector<int>{10, 20, 30});
    readme_add_kernel<<<2, 3>>>(values, rhs);
    synchronize();

    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{11, 22, 33, 4, 5}));
}

TEST(ReadmeExamples, AoSAndSoA)
{
    auto source = std::vector<Struct>{Struct{1, 2.0F, 3.0}, Struct{4, 5.0F, 6.0}, Struct{7, 8.0F, 9.0}};
    auto expected = std::vector<Struct>{Struct{2, 4.0F, 6.0}, Struct{8, 10.0F, 12.0}, Struct{14, 16.0F, 18.0}};

    auto aos = managed_array<Struct>(source);
    readme_soa_kernel<<<2, 3>>>(aos);
    synchronize();
    expect_records_eq(aos.template to<std::vector>(), expected);

    auto soa = managed_structure_of_arrays<Struct>(source);
    readme_soa_kernel<<<2, 3>>>(soa);
    synchronize();
    expect_records_eq(soa.template to<std::vector>(), expected);
}

TEST(ReadmeExamples, JaggedArray)
{
    auto flat = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto nested = std::vector<std::vector<int>>{{0}, {1, 2}, {3, 4, 5}, {6, 7, 8, 9}};

    auto from_nested = jagged_array<managed_array<int>>(nested);
    auto from_flat = jagged_array<managed_array<int>>({1, 2, 3, 4}, flat);

    readme_increment_kernel<<<2, 5>>>(from_nested);
    readme_increment_kernel<<<2, 5>>>(from_flat);
    synchronize();

    auto expected = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ(from_nested.template to<std::vector>(), expected);
    EXPECT_EQ(from_flat.template to<std::vector>(), expected);
    EXPECT_EQ((from_flat[{3U, 2U}]), 9);
}

TEST(ReadmeExamples, ZeroOverheadKernels)
{
    auto raw_pointer_values = managed_array<int, std::uint32_t>(8);
    auto array_values = managed_array<int, std::uint32_t>(8);
    auto view_values = managed_array<int, std::uint32_t>(8);

    readme_raw_pointer_kernel<<<1, 4>>>(raw_pointer_values.data(), raw_pointer_values.size());
    readme_array_kernel<<<1, 4>>>(array_values);
    readme_array_view_kernel<<<1, 4>>>(view_values);
    synchronize();

    const auto expected = std::vector<int>{1, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_EQ(raw_pointer_values.template to<std::vector>(), expected);
    EXPECT_EQ(array_values.template to<std::vector>(), expected);
    EXPECT_EQ(view_values.template to<std::vector>(), expected);
}

// NOLINTEND
