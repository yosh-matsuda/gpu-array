#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <vector>

// NOLINTBEGIN
namespace
{
    void synchronize() { GPU_CHECK_ERROR(gpu_array::api::gpuDeviceSynchronize()); }

    __global__ void grid_stride_blog_kernel(gpu_array::managed_array<int> array)
    {
        for (auto& value : array | gpu_array::views::grid_thread_stride)
        {
            value += 1;
        }
    }

    __global__ void enumerate_blog_kernel(gpu_array::managed_array<int> array)
    {
        for (auto&& [index, value] : array | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride)
        {
            value += static_cast<int>(index);
        }
    }

    __global__ void zip_blog_kernel(gpu_array::managed_array<int> lhs, gpu_array::managed_array<int> rhs)
    {
        for (auto&& [left, right] : gpu_array::views::zip(lhs, rhs) | gpu_array::views::grid_thread_stride)
        {
            left += right;
        }
    }

    template <std::ranges::input_range Range>
    requires std::ranges::input_range<std::ranges::range_value_t<Range>>
    __global__ void nested_blog_kernel(Range array)
    {
        for (auto& row : array | gpu_array::views::grid_block_stride)
        {
            for (auto& value : row | gpu_array::views::block_thread_stride)
            {
                value *= 2;
            }
        }
    }

    template <typename... Ts>
    requires (sizeof...(Ts) == 3)
    struct CustomTuple : public gpu_array::tuple<Ts...>
    {
        using base = gpu_array::tuple<Ts...>;
        using base::base;
        using base::operator=;

        __host__ __device__ decltype(auto) get_a() { return gpu_array::get<0>(*this); }
        __host__ __device__ decltype(auto) get_b() { return gpu_array::get<1>(*this); }
        __host__ __device__ decltype(auto) get_c() { return gpu_array::get<2>(*this); }
    };

    using Struct = CustomTuple<int, float, double>;

    template <typename Range>
    __global__ void soa_blog_kernel(Range array)
    {
        for (auto&& value : array | gpu_array::views::grid_thread_stride)
        {
            value.get_a() *= 2;
            value.get_b() *= 2.0F;
            value.get_c() *= 2.0;
        }
    }

    template <std::ranges::input_range Range>
    __global__ void jagged_blog_kernel(Range array)
    {
        for (auto& value : array | gpu_array::views::grid_thread_stride)
        {
            value *= 2;
        }
    }

    __global__ void managed_value_blog_kernel(gpu_array::managed_value<int> value) { *value += 7; }

    template <typename Record>
    void expect_records_eq(const std::vector<Record>& actual, const std::vector<Record>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (auto index = std::size_t{0}; index < actual.size(); ++index)
        {
            EXPECT_EQ(gpu_array::get<0>(actual[index]), gpu_array::get<0>(expected[index]));
            EXPECT_FLOAT_EQ(gpu_array::get<1>(actual[index]), gpu_array::get<1>(expected[index]));
            EXPECT_DOUBLE_EQ(gpu_array::get<2>(actual[index]), gpu_array::get<2>(expected[index]));
        }
    }
}  // namespace

TEST(BlogExamples, ManagedArrayConstruction)
{
    auto array = gpu_array::managed_array<int>(8);

    auto value = 0;
    for (auto& element : array)
    {
        element = value++;
    }

    EXPECT_EQ(array.template to<std::vector>(), (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(BlogExamples, GridStrideEnumerateAndZip)
{
    auto values = gpu_array::managed_array<int>(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7});

    grid_stride_blog_kernel<<<2, 5>>>(values);
    synchronize();
    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}));

    enumerate_blog_kernel<<<2, 5>>>(values);
    synchronize();
    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{1, 3, 5, 7, 9, 11, 13, 15}));

    auto rhs = gpu_array::managed_array<int>(std::vector<int>{10, 20, 30, 40, 50});
    zip_blog_kernel<<<2, 5>>>(values, rhs);
    synchronize();
    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{11, 23, 35, 47, 59, 11, 13, 15}));
}

TEST(BlogExamples, NestedArrayStride)
{
    auto vec_of_vec = std::vector(5, std::vector<int>(7, 21));
    auto array = gpu_array::managed_array(vec_of_vec);

    nested_blog_kernel<<<3, 4>>>(array);
    synchronize();

    auto expected = std::vector(5, std::vector<int>(7, 42));
    EXPECT_EQ(array.template to<std::vector>(), expected);
}

TEST(BlogExamples, AoSAndSoA)
{
    auto source = std::vector<Struct>{
        Struct{1, 2.0F, 3.0},
        Struct{4, 5.0F, 6.0},
        Struct{7, 8.0F, 9.0},
    };
    auto expected = std::vector<Struct>{
        Struct{2, 4.0F, 6.0},
        Struct{8, 10.0F, 12.0},
        Struct{14, 16.0F, 18.0},
    };

    auto aos = gpu_array::managed_array<Struct>(source);
    soa_blog_kernel<<<2, 3>>>(aos);
    synchronize();
    expect_records_eq(aos.template to<std::vector>(), expected);

    auto soa = gpu_array::managed_structure_of_arrays<Struct>(source);
    soa_blog_kernel<<<2, 3>>>(soa);
    synchronize();
    expect_records_eq(soa.template to<std::vector>(), expected);
}

TEST(BlogExamples, JaggedArray)
{
    auto flat = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto nested = std::vector<std::vector<int>>{{0}, {1, 2}, {3, 4, 5}, {6, 7, 8, 9}};

    auto from_nested = gpu_array::jagged_array<gpu_array::managed_array<int>>(nested);
    auto from_flat = gpu_array::jagged_array<gpu_array::managed_array<int>>({1, 2, 3, 4}, flat);

    jagged_blog_kernel<<<2, 5>>>(from_nested);
    jagged_blog_kernel<<<2, 5>>>(from_flat);
    synchronize();

    auto expected = std::vector<int>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    EXPECT_EQ(from_nested.template to<std::vector>(), expected);
    EXPECT_EQ(from_flat.template to<std::vector>(), expected);
    EXPECT_EQ(from_flat.num_rows(), 4U);
    EXPECT_EQ(from_flat.size(3), 4U);
    const auto index = std::array<typename decltype(from_flat)::size_type, 2>{3U, 2U};
    EXPECT_EQ(from_flat[index], 16);
}

TEST(BlogExamples, ManagedValue)
{
    auto value = gpu_array::managed_value<int>(35);

    managed_value_blog_kernel<<<1, 1>>>(value);
    synchronize();

    EXPECT_EQ(*value, 42);
}

// NOLINTEND
