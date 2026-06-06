#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

// NOLINTBEGIN
namespace
{
    using int_array = gpu_array::managed_array<int, std::uint32_t>;
    using grid_thread_view = gpu_array::detail::stride_view<gpu_array::detail::Stride::GridThread, int_array>;
    using enumerate_int_view = gpu_array::enumerate_view<int_array>;
    using zip_int_view = gpu_array::zip_view<int_array, int_array>;
    using grid_thread_strided_view = decltype(std::declval<int_array&>() | gpu_array::views::grid_thread_stride);
    using zip_then_enumerate_view =
        decltype(gpu_array::views::zip(std::declval<int_array&>(), std::declval<int_array&>()) |
                 gpu_array::views::enumerate);
    using enumerate_then_zip_view = decltype(gpu_array::views::zip(
        std::declval<int_array&>() | gpu_array::views::enumerate, std::declval<int_array&>()));
}  // namespace

TEST(StrideView, Concepts)
{
    static_assert(std::ranges::view<grid_thread_view>);
    static_assert(std::ranges::input_range<grid_thread_view>);
    static_assert(std::ranges::forward_range<grid_thread_view>);
    static_assert(!std::ranges::sized_range<grid_thread_view>);
    static_assert(std::same_as<std::ranges::range_value_t<grid_thread_view>, int>);
    static_assert(std::same_as<std::ranges::range_reference_t<grid_thread_view>, int&>);

#if !defined(ENABLE_HIP)
    using grid_block_alias = gpu_array::grid_block_stride_view<int_array>;
    using block_thread_alias = gpu_array::block_thread_stride_view<int_array>;
    static_assert(std::ranges::view<grid_block_alias>);
    static_assert(std::ranges::forward_range<grid_block_alias>);
    static_assert(std::ranges::view<block_thread_alias>);
    static_assert(std::ranges::forward_range<block_thread_alias>);
#endif

    SUCCEED();
}

TEST(StrideView, HostFallback)
{
    auto values = int_array(std::vector<int>{0, 1, 2, 3, 4, 5, 6});

    for (auto& value : values | gpu_array::views::grid_thread_stride)
    {
        value += 10;
    }
    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{10, 11, 12, 13, 14, 15, 16}));

    for (auto& value : gpu_array::views::block_thread_stride(values))
    {
        value *= 2;
    }
    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{20, 22, 24, 26, 28, 30, 32}));
}

TEST(ViewComposition, StrideOrderConstraints)
{
    static_assert(!std::ranges::sized_range<grid_thread_strided_view>);
    static_assert(
        requires(int_array& values) { values | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride; });
    static_assert(requires(int_array& lhs, int_array& rhs) {
        gpu_array::views::zip(lhs, rhs) | gpu_array::views::grid_thread_stride;
    });
    static_assert(!std::invocable<decltype(gpu_array::views::enumerate), grid_thread_strided_view&>);
    static_assert(
        !std::invocable<decltype(gpu_array::views::zip), grid_thread_strided_view&, grid_thread_strided_view&>);

    SUCCEED();
}

TEST(EnumerateView, Concepts)
{
    static_assert(std::ranges::view<enumerate_int_view>);
    static_assert(std::ranges::sized_range<enumerate_int_view>);
    static_assert(std::ranges::input_range<enumerate_int_view>);
    static_assert(std::ranges::forward_range<enumerate_int_view>);
    static_assert(std::ranges::bidirectional_range<enumerate_int_view>);
    static_assert(std::ranges::random_access_range<enumerate_int_view>);
    static_assert(std::same_as<std::ranges::range_value_t<enumerate_int_view>, gpu_array::tuple<std::uint32_t, int>>);

    SUCCEED();
}

TEST(EnumerateView, Indices)
{
    auto values = int_array(std::vector<int>{4, 5, 6, 7});
    auto enumerated = values | gpu_array::views::enumerate;

    EXPECT_EQ(enumerated.size(), 4U);
    EXPECT_FALSE(enumerated.empty());

    auto first = enumerated.front();
    EXPECT_EQ(gpu_array::get<0>(first), 0U);
    EXPECT_EQ(gpu_array::get<1>(first), 4);
    gpu_array::get<1>(first) = 40;

    auto third = enumerated[2];
    EXPECT_EQ(gpu_array::get<0>(third), 2U);
    EXPECT_EQ(gpu_array::get<1>(third), 6);
    gpu_array::get<1>(third) = 60;

    auto expected = std::vector<int>{40, 5, 60, 7};
    EXPECT_EQ(values.template to<std::vector>(), expected);

    auto indexes = std::vector<std::uint32_t>{};
    auto seen = std::vector<int>{};
    for (auto&& [index, value] : values | gpu_array::views::enumerate)
    {
        indexes.push_back(index);
        seen.push_back(value);
    }
    EXPECT_EQ(indexes, (std::vector<std::uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(seen, expected);
}

TEST(ZipView, Concepts)
{
    static_assert(std::ranges::view<zip_int_view>);
    static_assert(std::ranges::sized_range<zip_int_view>);
    static_assert(std::ranges::input_range<zip_int_view>);
    static_assert(std::ranges::forward_range<zip_int_view>);
    static_assert(std::same_as<std::ranges::range_value_t<zip_int_view>, gpu_array::tuple<int, int>>);

    SUCCEED();
}

TEST(ZipView, ShortestRange)
{
    auto lhs = int_array(std::vector<int>{1, 2, 3, 4, 5});
    auto rhs = int_array(std::vector<int>{10, 20, 30});
    auto zipped = gpu_array::views::zip(lhs, rhs);

    EXPECT_EQ(zipped.size(), 3U);
    EXPECT_FALSE(zipped.empty());

    auto first = zipped.front();
    EXPECT_EQ(gpu_array::get<0>(first), 1);
    EXPECT_EQ(gpu_array::get<1>(first), 10);

    auto second = zipped[1];
    gpu_array::get<0>(second) += gpu_array::get<1>(second);

    for (auto&& [left, right] : zipped)
    {
        left += right;
    }

    EXPECT_EQ(lhs.template to<std::vector>(), (std::vector<int>{11, 42, 33, 4, 5}));
    EXPECT_EQ(rhs.template to<std::vector>(), (std::vector<int>{10, 20, 30}));
}

TEST(ZipView, EmptyRange)
{
    auto lhs = int_array(std::vector<int>{1, 2, 3});
    auto rhs = int_array();
    auto zipped = gpu_array::views::zip(lhs, rhs);

    EXPECT_EQ(zipped.size(), 0U);
    EXPECT_TRUE(zipped.empty());

    for (auto&& [left, right] : zipped)
    {
        left += right;
    }
    EXPECT_EQ(lhs.template to<std::vector>(), (std::vector<int>{1, 2, 3}));
}

TEST(ViewComposition, ZipEnumerateOrder)
{
    static_assert(std::same_as<std::ranges::range_reference_t<zip_then_enumerate_view>,
                               gpu_array::tuple<std::uint32_t, gpu_array::tuple<int&, int&>>>);
    static_assert(std::same_as<std::ranges::range_reference_t<enumerate_then_zip_view>,
                               gpu_array::tuple<gpu_array::tuple<std::uint32_t, int&>, int&>>);

    auto lhs = int_array(std::vector<int>{1, 2, 3, 4});
    auto rhs = int_array(std::vector<int>{10, 20, 30});

    auto zip_then_enumerate = gpu_array::views::zip(lhs, rhs) | gpu_array::views::enumerate;
    auto first = zip_then_enumerate[0];
    EXPECT_EQ(gpu_array::get<0>(first), 0U);
    auto first_pair = gpu_array::get<1>(first);
    EXPECT_EQ(gpu_array::get<0>(first_pair), 1);
    EXPECT_EQ(gpu_array::get<1>(first_pair), 10);

    auto enumerate_then_zip = gpu_array::views::zip(lhs | gpu_array::views::enumerate, rhs);
    auto second = enumerate_then_zip[1];
    auto second_enumerated = gpu_array::get<0>(second);
    EXPECT_EQ(gpu_array::get<0>(second_enumerated), 1U);
    EXPECT_EQ(gpu_array::get<1>(second_enumerated), 2);
    EXPECT_EQ(gpu_array::get<1>(second), 20);

    gpu_array::get<0>(first_pair) = 100;
    gpu_array::get<1>(second_enumerated) = 200;
    gpu_array::get<1>(second) = 2000;

    EXPECT_EQ(lhs.template to<std::vector>(), (std::vector<int>{100, 200, 3, 4}));
    EXPECT_EQ(rhs.template to<std::vector>(), (std::vector<int>{10, 2000, 30}));
}

// NOLINTEND
