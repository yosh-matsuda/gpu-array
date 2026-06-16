#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

// NOLINTBEGIN
namespace
{
    using managed_int_array = gpu_array::managed_array<int, std::uint32_t>;
    using managed_int_jagged = gpu_array::jagged_array<managed_int_array>;

    template <typename... Ts>
    requires (sizeof...(Ts) == 3)
    struct gpu_tuple_record : public gpu_array::tuple<Ts...>
    {
        using base = gpu_array::tuple<Ts...>;
        using base::base;
        using base::operator=;

        __host__ __device__ decltype(auto) get_a() { return gpu_array::get<0>(*this); }
        __host__ __device__ decltype(auto) get_b() { return gpu_array::get<1>(*this); }
        __host__ __device__ decltype(auto) get_c() { return gpu_array::get<2>(*this); }

        __host__ __device__ decltype(auto) get_a() const { return gpu_array::get<0>(*this); }
        __host__ __device__ decltype(auto) get_b() const { return gpu_array::get<1>(*this); }
        __host__ __device__ decltype(auto) get_c() const { return gpu_array::get<2>(*this); }
    };

    template <typename... Ts>
    requires (sizeof...(Ts) == 3)
    struct std_tuple_record : public std::tuple<Ts...>
    {
        using base = std::tuple<Ts...>;
        using base::base;
        using base::operator=;

        __host__ __device__ decltype(auto) get_a() { return std::get<0>(*this); }
        __host__ __device__ decltype(auto) get_b() { return std::get<1>(*this); }
        __host__ __device__ decltype(auto) get_c() { return std::get<2>(*this); }

        __host__ __device__ decltype(auto) get_a() const { return std::get<0>(*this); }
        __host__ __device__ decltype(auto) get_b() const { return std::get<1>(*this); }
        __host__ __device__ decltype(auto) get_c() const { return std::get<2>(*this); }
    };

    using gpu_record = gpu_tuple_record<int, float, double>;
    using std_record = std_tuple_record<int, float, double>;

    auto make_jagged_ints() { return std::vector<std::vector<int>>{{1}, {2, 3}, {}, {4, 5, 6}}; }

    template <typename Jagged>
    constexpr void check_jagged_range_concepts()
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        static_assert(std::ranges::range<Jagged>);
#if defined(GPU_DEVICE_COMPILE)
        static_assert(std::ranges::borrowed_range<Jagged>);
#else
        static_assert(!std::ranges::borrowed_range<Jagged>);
#endif
        static_assert(std::ranges::view<Jagged>);
        static_assert(std::ranges::input_range<Jagged>);
        static_assert(std::ranges::forward_range<Jagged>);
        static_assert(std::ranges::bidirectional_range<Jagged>);
        static_assert(std::ranges::random_access_range<Jagged>);
        static_assert(std::ranges::sized_range<Jagged>);
        static_assert(std::ranges::common_range<Jagged>);
        static_assert(std::ranges::viewable_range<Jagged>);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    }

    std::vector<int> row_to_vector(const managed_int_jagged& value, std::uint32_t row)
    {
        const auto row_view = value.row(row);
        return {row_view.begin(), row_view.end()};
    }

    template <typename Record>
    void expect_record_eq(const Record& actual, int a, float b, double c)
    {
        EXPECT_EQ(actual.get_a(), a);
        EXPECT_FLOAT_EQ(actual.get_b(), b);
        EXPECT_DOUBLE_EQ(actual.get_c(), c);
    }

    template <typename Record>
    void expect_records_eq(const std::vector<Record>& actual, const std::vector<Record>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (auto index = std::size_t{0}; index < actual.size(); ++index)
        {
            EXPECT_EQ(actual[index].get_a(), expected[index].get_a());
            EXPECT_FLOAT_EQ(actual[index].get_b(), expected[index].get_b());
            EXPECT_DOUBLE_EQ(actual[index].get_c(), expected[index].get_c());
        }
    }
}  // namespace

namespace std
{
    template <class... TTypes, class... UTypes>
    requires requires { typename tuple<common_type_t<TTypes, UTypes>...>; }
    struct common_type<std_tuple_record<TTypes...>, std_tuple_record<UTypes...>>
    {
        using type = tuple<common_type_t<TTypes, UTypes>...>;
    };

    template <class... TTypes, class... UTypes, template <class> class TQual, template <class> class UQual>
    requires requires {
        typename tuple<common_reference_t<const remove_reference_t<TTypes>&, const remove_reference_t<UTypes>&>...>;
    }
    struct basic_common_reference<std_tuple_record<TTypes...>, std_tuple_record<UTypes...>, TQual, UQual>
    {
        using type = tuple<common_reference_t<const remove_reference_t<TTypes>&, const remove_reference_t<UTypes>&>...>;
    };
}  // namespace std

TEST(JaggedArrayTypes, Constraints)
{
    using managed_gpu_record_soa = gpu_array::managed_structure_of_arrays<gpu_record, std::uint32_t>;
    using managed_gpu_record_jagged = gpu_array::jagged_array<managed_gpu_record_soa>;
    using managed_std_record_soa = gpu_array::managed_structure_of_arrays<std_record, std::uint32_t>;
    using managed_std_record_jagged = gpu_array::jagged_array<managed_std_record_soa>;
    using enumerate_int_jagged_view = gpu_array::enumerate_view<managed_int_jagged>;
    using zip_int_jagged_view = gpu_array::zip_view<managed_int_jagged, managed_int_jagged>;
    using const_zip_int_jagged_view =
        decltype(gpu_array::views::zip(std::declval<managed_int_jagged&>(), std::declval<const managed_int_jagged&>()));
    using enumerate_record_jagged_view = gpu_array::enumerate_view<managed_gpu_record_jagged>;
    using zip_record_jagged_view = gpu_array::zip_view<managed_gpu_record_jagged, managed_gpu_record_jagged>;
    using const_zip_record_jagged_view = decltype(gpu_array::views::zip(
        std::declval<managed_gpu_record_jagged&>(), std::declval<const managed_gpu_record_jagged&>()));

    static_assert(gpu_array::gpu_managed_random_access_range<managed_int_array>);
    static_assert(gpu_array::gpu_managed_random_access_range<managed_gpu_record_soa>);
    static_assert(gpu_array::gpu_managed_random_access_range<managed_std_record_soa>);
    static_assert(!gpu_array::gpu_managed_random_access_range<gpu_array::array<int, std::uint32_t>>);
    static_assert(!gpu_array::gpu_managed_random_access_range<gpu_array::managed_value<int>>);

    static_assert(std::same_as<typename managed_int_jagged::size_type, std::uint32_t>);
    static_assert(std::same_as<std::ranges::range_value_t<managed_int_jagged>, int>);
    static_assert(std::same_as<std::ranges::range_value_t<managed_gpu_record_jagged>, gpu_record>);
    static_assert(std::same_as<std::ranges::range_value_t<managed_std_record_jagged>, std_record>);
    static_assert(std::same_as<decltype(std::declval<managed_gpu_record_jagged&>().template data<0>()), int*>);
    static_assert(
        std::same_as<decltype(std::declval<const managed_gpu_record_jagged&>().template data<0>()), const int*>);
    static_assert(std::same_as<decltype(std::declval<managed_std_record_jagged&>().template data<0>()), int*>);
    static_assert(
        std::same_as<decltype(std::declval<const managed_std_record_jagged&>().template data<0>()), const int*>);
    static_assert(std::ranges::input_range<enumerate_int_jagged_view>);
    static_assert(
        std::same_as<std::ranges::range_value_t<enumerate_int_jagged_view>, gpu_array::tuple<std::uint32_t, int>>);
    static_assert(
        std::same_as<std::ranges::range_reference_t<enumerate_int_jagged_view>, gpu_array::tuple<std::uint32_t, int&>>);
    static_assert(std::ranges::input_range<zip_int_jagged_view>);
    static_assert(std::same_as<std::ranges::range_value_t<zip_int_jagged_view>, gpu_array::tuple<int, int>>);
    static_assert(std::same_as<std::ranges::range_reference_t<zip_int_jagged_view>, gpu_array::tuple<int&, int&>>);
    static_assert(
        std::same_as<std::ranges::range_reference_t<const_zip_int_jagged_view>, gpu_array::tuple<int&, const int&>>);
    static_assert(std::ranges::input_range<enumerate_record_jagged_view>);
    static_assert(std::same_as<std::ranges::range_value_t<enumerate_record_jagged_view>,
                               gpu_array::tuple<std::uint32_t, gpu_record>>);
    static_assert(std::same_as<std::ranges::range_reference_t<enumerate_record_jagged_view>,
                               gpu_array::tuple<std::uint32_t, gpu_tuple_record<int&, float&, double&>>>);
    static_assert(std::ranges::input_range<zip_record_jagged_view>);
    static_assert(
        std::same_as<std::ranges::range_value_t<zip_record_jagged_view>, gpu_array::tuple<gpu_record, gpu_record>>);
    static_assert(std::same_as<
                  std::ranges::range_reference_t<zip_record_jagged_view>,
                  gpu_array::tuple<gpu_tuple_record<int&, float&, double&>, gpu_tuple_record<int&, float&, double&>>>);
    static_assert(std::same_as<std::ranges::range_reference_t<const_zip_record_jagged_view>,
                               gpu_array::tuple<gpu_tuple_record<int&, float&, double&>,
                                                gpu_tuple_record<const int&, const float&, const double&>>>);
    static_assert(requires(managed_int_jagged& values) {
        values | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride;
    });
    static_assert(requires(managed_int_jagged& lhs, managed_int_jagged& rhs) {
        gpu_array::views::zip(lhs, rhs) | gpu_array::views::grid_thread_stride;
    });
    static_assert(requires(managed_gpu_record_jagged& values) {
        values | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride;
    });
    static_assert(requires(managed_gpu_record_jagged& lhs, managed_gpu_record_jagged& rhs) {
        gpu_array::views::zip(lhs, rhs) | gpu_array::views::grid_thread_stride;
    });

    check_jagged_range_concepts<managed_int_jagged>();
    check_jagged_range_concepts<managed_gpu_record_jagged>();
    check_jagged_range_concepts<managed_std_record_jagged>();

    SUCCEED();
}

TEST(JaggedArray, EmptyAndSizes)
{
    auto default_value = managed_int_jagged();
    EXPECT_EQ(default_value.size(), 0U);
    EXPECT_TRUE(default_value.empty());
    EXPECT_EQ(default_value.num_rows(), 0U);
    EXPECT_EQ(default_value.data(), nullptr);

    auto sized = managed_int_jagged({1U, 0U, 3U});
    EXPECT_EQ(sized.size(), 4U);
    EXPECT_EQ(sized.num_rows(), 3U);
    EXPECT_EQ(sized.size(0), 1U);
    EXPECT_EQ(sized.size(1), 0U);
    EXPECT_EQ(sized.size(2), 3U);
}

TEST(JaggedArray, NestedRows)
{
    const auto source = std::vector<std::vector<int>>{{0}, {}, {1, 2, 3}, {4, 5}};
    auto value = managed_int_jagged(source);

    EXPECT_EQ(value.size(), 6U);
    EXPECT_EQ(value.num_rows(), 4U);
    EXPECT_EQ(row_to_vector(value, 0), (std::vector<int>{0}));
    EXPECT_TRUE(row_to_vector(value, 1).empty());
    EXPECT_EQ(row_to_vector(value, 2), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(row_to_vector(value, 3), (std::vector<int>{4, 5}));

    EXPECT_EQ(value.begin(1), value.end(1));
    EXPECT_EQ(value.data(1), value.data() + 1);
    EXPECT_EQ(value.data(1), value.data(2));

    EXPECT_EQ((value[{0U, 0U}]), 0);
    EXPECT_EQ((value[{2U, 1U}]), 2);
    EXPECT_EQ((value[{3U, 1U}]), 5);
    EXPECT_EQ(value.data(2)[0], 1);
    EXPECT_EQ(value.data(2)[2], 3);
}

TEST(JaggedArray, FlatRows)
{
    const auto flat = std::vector<int>{0, 1, 2, 3, 4, 5};
    auto value = managed_int_jagged({1U, 2U, 3U}, flat);

    EXPECT_EQ(value.template to<std::vector>(), flat);
    EXPECT_EQ(value.num_rows(), 3U);
    EXPECT_EQ(row_to_vector(value, 0), (std::vector<int>{0}));
    EXPECT_EQ(row_to_vector(value, 1), (std::vector<int>{1, 2}));
    EXPECT_EQ(row_to_vector(value, 2), (std::vector<int>{3, 4, 5}));

    EXPECT_THROW((managed_int_jagged({1U, 2U}, flat)), std::invalid_argument);
}

TEST(JaggedArray, SharedBase)
{
    auto base = managed_int_array(std::vector<int>{10, 20, 30, 40, 50});
    auto value = managed_int_jagged(std::array<std::uint32_t, 2>{2U, 3U}, base);

    EXPECT_EQ(value.data(), base.data());
    EXPECT_EQ(base.use_count(), 2U);
    EXPECT_EQ(value.use_count(), 2U);

    base[1] = 21;
    value[{1U, 2U}] = 55;
    EXPECT_EQ((value[{0U, 1U}]), 21);
    EXPECT_EQ(base[4], 55);
}

TEST(JaggedArray, GpuTupleSoa)
{
    using record = gpu_record;
    using managed_record_soa = gpu_array::managed_structure_of_arrays<record, std::uint32_t>;
    using managed_record_jagged = gpu_array::jagged_array<managed_record_soa>;

    const auto nested = std::vector<std::vector<record>>{
        {record{1, 1.5F, 2.5}},
        {record{2, 3.5F, 4.5}, record{3, 5.5F, 6.5}},
        {},
        {record{4, 7.5F, 8.5}},
    };
    auto value = managed_record_jagged(nested);

    EXPECT_EQ(value.size(), 4U);
    EXPECT_EQ(value.num_rows(), 4U);
    EXPECT_EQ(value.size(0), 1U);
    EXPECT_EQ(value.size(1), 2U);
    EXPECT_EQ(value.size(2), 0U);
    EXPECT_EQ(value.size(3), 1U);

    expect_record_eq(value[{0U, 0U}], 1, 1.5F, 2.5);
    expect_record_eq(value[{1U, 1U}], 3, 5.5F, 6.5);
    expect_record_eq(value[{3U, 0U}], 4, 7.5F, 8.5);

    value[{1U, 0U}].get_a() = 20;
    value[{1U, 0U}].get_b() = 30.0F;
    value[{1U, 0U}].get_c() = 40.0;
    EXPECT_EQ(value.template data<0>()[1], 20);
    EXPECT_FLOAT_EQ(value.template data<1>()[1], 30.0F);
    EXPECT_DOUBLE_EQ(value.template data<2>()[1], 40.0);
    EXPECT_EQ(value.template data<0>(1)[0], 20);
    EXPECT_EQ(value.template data<0>(1)[1], 3);
    EXPECT_FLOAT_EQ(value.template data<1>(3)[0], 7.5F);
    EXPECT_DOUBLE_EQ(value.template data<2>(3)[0], 8.5);
}

TEST(JaggedArray, FlatViews)
{
    auto values = managed_int_jagged(make_jagged_ints());

    for (auto&& [index, value] : values | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride)
    {
        value += static_cast<int>(index) * 10;
    }

    auto rhs = managed_int_jagged(make_jagged_ints());
    for (auto&& [left, right] : gpu_array::views::zip(values, rhs) | gpu_array::views::grid_thread_stride)
    {
        left += right;
    }

    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{2, 14, 26, 38, 50, 62}));
}

TEST(JaggedArray, RowViews)
{
    auto values = managed_int_jagged(make_jagged_ints());

    for (auto row_index = std::size_t{0}; row_index < values.num_rows(); ++row_index)
    {
        for (auto&& [column_index, value] :
             values.row(row_index) | gpu_array::views::enumerate | gpu_array::views::block_thread_stride)
        {
            value += static_cast<int>(row_index * 100 + column_index);
        }
    }

    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{1, 102, 104, 304, 306, 308}));
}

TEST(JaggedArray, GpuTupleSoaFlatViews)
{
    using record = gpu_record;
    using managed_record_soa = gpu_array::managed_structure_of_arrays<record, std::uint32_t>;
    using managed_record_jagged = gpu_array::jagged_array<managed_record_soa>;

    const auto nested = std::vector<std::vector<record>>{
        {record{1, 1.5F, 2.5}},
        {record{2, 2.5F, 3.5}, record{3, 3.5F, 4.5}},
        {},
        {record{4, 4.5F, 5.5}},
    };
    auto records = managed_record_jagged(nested);

    for (auto&& [index, value] : records | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride)
    {
        value.get_a() += static_cast<int>(index) * 10;
    }

    auto rhs = managed_record_jagged(nested);
    for (auto&& [left, right] : gpu_array::views::zip(records, rhs) | gpu_array::views::grid_thread_stride)
    {
        left.get_b() += right.get_b();
        left.get_c() += right.get_c();
    }

    expect_records_eq(records.template to<std::vector>(),
                      std::vector<record>{record{1, 3.0F, 5.0}, record{12, 5.0F, 7.0}, record{23, 7.0F, 9.0},
                                          record{34, 9.0F, 11.0}});
}

TEST(JaggedArray, StdTupleSoa)
{
    using record = std_record;
    using managed_record_soa = gpu_array::managed_structure_of_arrays<record, std::uint32_t>;
    using managed_record_jagged = gpu_array::jagged_array<managed_record_soa>;

    const auto nested = std::vector<std::vector<record>>{
        {record{1, 1.5F, 2.5}},
        {record{2, 3.5F, 4.5}, record{3, 5.5F, 6.5}},
        {},
        {record{4, 7.5F, 8.5}},
    };
    auto value = managed_record_jagged(nested);

    EXPECT_EQ(value.size(), 4U);
    EXPECT_EQ(value.num_rows(), 4U);
    EXPECT_EQ(value.size(0), 1U);
    EXPECT_EQ(value.size(1), 2U);
    EXPECT_EQ(value.size(2), 0U);
    EXPECT_EQ(value.size(3), 1U);

    expect_record_eq(value[{0U, 0U}], 1, 1.5F, 2.5);
    expect_record_eq(value[{1U, 1U}], 3, 5.5F, 6.5);
    expect_record_eq(value[{3U, 0U}], 4, 7.5F, 8.5);

    value[{1U, 0U}].get_a() = 20;
    value[{1U, 0U}].get_b() = 30.0F;
    value[{1U, 0U}].get_c() = 40.0;
    EXPECT_EQ(value.template data<0>()[1], 20);
    EXPECT_FLOAT_EQ(value.template data<1>()[1], 30.0F);
    EXPECT_DOUBLE_EQ(value.template data<2>()[1], 40.0);
    EXPECT_EQ(value.template data<0>(1)[0], 20);
    EXPECT_EQ(value.template data<0>(1)[1], 3);
    EXPECT_FLOAT_EQ(value.template data<1>(3)[0], 7.5F);
    EXPECT_DOUBLE_EQ(value.template data<2>(3)[0], 8.5);
}

// NOLINTEND
