#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

// NOLINTBEGIN
using namespace gpu_array;

namespace
{
    struct no_default
    {
        no_default() = delete;
        explicit no_default(int value_) : value(value_) {}
        int value;
    };

    struct move_observer
    {
        int value = 0;
        bool moved_from = false;

        __host__ __device__ move_observer() {}
        __host__ __device__ explicit move_observer(int value_) : value(value_) {}
        __host__ __device__ move_observer(const move_observer& other) : value(other.value), moved_from(other.moved_from)
        {
        }
        __host__ __device__ move_observer(move_observer&& other) noexcept
            : value(other.value), moved_from(other.moved_from)
        {
            other.moved_from = true;
        }
        __host__ __device__ move_observer& operator=(move_observer&& other) noexcept
        {
            value = other.value;
            moved_from = other.moved_from;
            other.moved_from = true;
            return *this;
        }
    };

    struct call_marker
    {
        bool* called;

        __host__ __device__ void operator()(int, double) const { *called = true; }
    };

    struct index_list
    {
    };

    struct derived_record : gpu_array::tuple<float, unsigned>
    {
        using base = gpu_array::tuple<float, unsigned>;

        derived_record(float value, const index_list&) : base(value, 7U), used_custom_constructor(true) {}

        using base::base;

        bool used_custom_constructor = false;
    };

    struct weighted_char_sum
    {
        __host__ __device__ double operator()(int i, double d, char c) const { return i + d + (c == 'x' ? 10.0 : 0.0); }
    };

    struct mutate_tuple_values
    {
        __host__ __device__ void operator()(int& i, double& d, char& c) const
        {
            i += 1;
            d += 1.0;
            c = 'y';
        }
    };

    struct sum_const_values
    {
        __host__ __device__ double operator()(const int& i, const double& d) const { return i + d; }
    };

    struct move_observer_value
    {
        __host__ __device__ move_observer operator()(move_observer&& observer) const
        {
            return move_observer(std::move(observer));
        }
    };

    template <typename... Ts>
    requires (sizeof...(Ts) == 3)
    struct custom_tuple : public gpu_array::tuple<Ts...>
    {
        using base = gpu_array::tuple<Ts...>;
        using base::base;
        using base::operator=;

        template <typename... Us>
        __host__ __device__ custom_tuple(const custom_tuple<Us...>& other) : base(other)
        {
        }

        __host__ __device__ decltype(auto) get_a() { return gpu_array::get<0>(*this); }
        __host__ __device__ decltype(auto) get_b() { return gpu_array::get<1>(*this); }
        __host__ __device__ decltype(auto) get_c() { return gpu_array::get<2>(*this); }

        __host__ __device__ decltype(auto) get_a() const { return gpu_array::get<0>(*this); }
        __host__ __device__ decltype(auto) get_b() const { return gpu_array::get<1>(*this); }
        __host__ __device__ decltype(auto) get_c() const { return gpu_array::get<2>(*this); }
    };

    template <typename Tuple>
    constexpr bool has_unqualified_get()
    {
        using gpu_array::get;
        return requires(Tuple value) {
            get<0>(value);
            get<1>(value);
        };
    }

    template <typename Tuple>
    auto first_two_sum(Tuple& value)
    {
        using gpu_array::get;
        return get<0>(value) + get<1>(value);
    }

    template <std::size_t I, typename GpuTuple, typename StdTuple>
    constexpr bool index_get_matches_std_get()
    {
        return std::same_as<decltype(gpu_array::get<I>(std::declval<GpuTuple&>())),
                            decltype(std::get<I>(std::declval<StdTuple&>()))> &&
               (noexcept(gpu_array::get<I>(std::declval<GpuTuple&>())) ==
                noexcept(std::get<I>(std::declval<StdTuple&>()))) &&
               std::same_as<decltype(gpu_array::get<I>(std::declval<const GpuTuple&>())),
                            decltype(std::get<I>(std::declval<const StdTuple&>()))> &&
               (noexcept(gpu_array::get<I>(std::declval<const GpuTuple&>())) ==
                noexcept(std::get<I>(std::declval<const StdTuple&>()))) &&
               std::same_as<decltype(gpu_array::get<I>(std::declval<GpuTuple&&>())),
                            decltype(std::get<I>(std::declval<StdTuple&&>()))> &&
               (noexcept(gpu_array::get<I>(std::declval<GpuTuple&&>())) ==
                noexcept(std::get<I>(std::declval<StdTuple&&>()))) &&
               std::same_as<decltype(gpu_array::get<I>(std::declval<const GpuTuple&&>())),
                            decltype(std::get<I>(std::declval<const StdTuple&&>()))> &&
               (noexcept(gpu_array::get<I>(std::declval<const GpuTuple&&>())) ==
                noexcept(std::get<I>(std::declval<const StdTuple&&>())));
    }

    template <typename T, typename GpuTuple, typename StdTuple>
    constexpr bool type_get_matches_std_get()
    {
        return std::same_as<decltype(gpu_array::get<T>(std::declval<GpuTuple&>())),
                            decltype(std::get<T>(std::declval<StdTuple&>()))> &&
               (noexcept(gpu_array::get<T>(std::declval<GpuTuple&>())) ==
                noexcept(std::get<T>(std::declval<StdTuple&>()))) &&
               std::same_as<decltype(gpu_array::get<T>(std::declval<const GpuTuple&>())),
                            decltype(std::get<T>(std::declval<const StdTuple&>()))> &&
               (noexcept(gpu_array::get<T>(std::declval<const GpuTuple&>())) ==
                noexcept(std::get<T>(std::declval<const StdTuple&>()))) &&
               std::same_as<decltype(gpu_array::get<T>(std::declval<GpuTuple&&>())),
                            decltype(std::get<T>(std::declval<StdTuple&&>()))> &&
               (noexcept(gpu_array::get<T>(std::declval<GpuTuple&&>())) ==
                noexcept(std::get<T>(std::declval<StdTuple&&>()))) &&
               std::same_as<decltype(gpu_array::get<T>(std::declval<const GpuTuple&&>())),
                            decltype(std::get<T>(std::declval<const StdTuple&&>()))> &&
               (noexcept(gpu_array::get<T>(std::declval<const GpuTuple&&>())) ==
                noexcept(std::get<T>(std::declval<const StdTuple&&>())));
    }
}  // namespace

template <class... TTypes, class... UTypes>
requires requires { typename custom_tuple<std::common_type_t<TTypes, UTypes>...>; }
struct std::common_type<custom_tuple<TTypes...>, custom_tuple<UTypes...>>
{
    using type = custom_tuple<std::common_type_t<TTypes, UTypes>...>;
};

template <class... TTypes, class... UTypes, template <class> class TQual, template <class> class UQual>
requires requires { typename custom_tuple<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>; }
struct std::basic_common_reference<custom_tuple<TTypes...>, custom_tuple<UTypes...>, TQual, UQual>
{
    using type = custom_tuple<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>;
};

TEST(Tuple, TypeTraits)
{
    using empty_tuple = gpu_array::tuple<>;
    using tuple_type = gpu_array::tuple<int, double, char>;
    using cv_tuple_type = gpu_array::tuple<const int, volatile double>;

    static_assert(std::tuple_size_v<empty_tuple> == 0);
    static_assert(std::tuple_size_v<tuple_type> == 3);
    static_assert(std::same_as<std::tuple_element_t<0, tuple_type>, int>);
    static_assert(std::same_as<std::tuple_element_t<1, tuple_type>, double>);
    static_assert(std::same_as<std::tuple_element_t<2, tuple_type>, char>);
    static_assert(std::same_as<std::tuple_element_t<0, cv_tuple_type>, const int>);
    static_assert(std::same_as<std::tuple_element_t<1, cv_tuple_type>, volatile double>);

    auto deduced = gpu_array::tuple(1, 2.5);
    static_assert(std::same_as<decltype(deduced), gpu_array::tuple<int, double>>);

    static_assert(std::default_initializable<gpu_array::tuple<>>);
    static_assert(std::default_initializable<gpu_array::tuple<int, double>>);
    static_assert(!std::default_initializable<gpu_array::tuple<no_default>>);
    static_assert(std::constructible_from<gpu_array::tuple<no_default>, int>);
    static_assert(std::constructible_from<gpu_array::tuple<double, long>, const gpu_array::tuple<int, int>&>);

    EXPECT_EQ(std::tuple_size_v<tuple_type>, 3U);
}

TEST(Tuple, ConstructionAndElementAccess)
{
    static_assert(std::tuple_size_v<gpu_array::tuple<>> == 0);

    auto single = gpu_array::tuple<int>(7);
    EXPECT_EQ(gpu_array::get<0>(single), 7);
    gpu_array::get<0>(single) = 11;
    EXPECT_EQ(gpu_array::get<0>(single), 11);

    auto value = gpu_array::tuple<int, double, char>(1, 2.5, 'g');
    static_assert(std::same_as<decltype(gpu_array::get<0>(value)), int&>);
    static_assert(std::same_as<decltype(gpu_array::get<1>(std::as_const(value))), const double&>);
    static_assert(std::same_as<decltype(gpu_array::get<2>(std::move(value))), char&&>);

    EXPECT_EQ(gpu_array::get<0>(value), 1);
    EXPECT_EQ(gpu_array::get<1>(value), 2.5);
    EXPECT_EQ(gpu_array::get<2>(value), 'g');

    const auto const_value = gpu_array::tuple<int, double>(3, 4.5);
    static_assert(std::same_as<decltype(gpu_array::get<0>(std::move(const_value))), const int&&>);
    EXPECT_EQ(gpu_array::get<0>(const_value), 3);
    EXPECT_EQ(gpu_array::get<1>(const_value), 4.5);
}

TEST(Tuple, DerivedConstructorWinsWhenInheritedBaseIsNotConstructible)
{
    static_assert(!std::constructible_from<derived_record::base, float, const index_list&>);
    static_assert(std::constructible_from<derived_record, float, const index_list&>);

    const index_list indices;
    const auto record = derived_record(1.0F, indices);

    EXPECT_TRUE(record.used_custom_constructor);
    EXPECT_FLOAT_EQ(gpu_array::get<0>(record), 1.0F);
    EXPECT_EQ(gpu_array::get<1>(record), 7U);
}

TEST(Tuple, GetMatchesStdGet)
{
    static_assert(index_get_matches_std_get<0, gpu_array::tuple<int>, std::tuple<int>>());
    static_assert(index_get_matches_std_get<0, gpu_array::tuple<int*>, std::tuple<int*>>());
    static_assert(index_get_matches_std_get<0, gpu_array::tuple<int&>, std::tuple<int&>>());
    static_assert(index_get_matches_std_get<0, gpu_array::tuple<int&&>, std::tuple<int&&>>());

    static_assert(type_get_matches_std_get<int, gpu_array::tuple<int, double>, std::tuple<int, double>>());
    static_assert(type_get_matches_std_get<double, gpu_array::tuple<int, double>, std::tuple<int, double>>());
    static_assert(type_get_matches_std_get<int&, gpu_array::tuple<int&, double>, std::tuple<int&, double>>());
    static_assert(type_get_matches_std_get<int&&, gpu_array::tuple<int&&, double>, std::tuple<int&&, double>>());

    auto value = gpu_array::tuple<int, double, char>(1, 2.5, 'x');
    gpu_array::get<double>(value) = 3.5;
    EXPECT_EQ(gpu_array::get<int>(value), 1);
    EXPECT_EQ(gpu_array::get<double>(value), 3.5);
    EXPECT_EQ(gpu_array::get<char>(std::move(value)), 'x');

    auto target = 4;
    auto referenced = gpu_array::tuple<int&>(target);
    const auto& const_referenced = referenced;
    gpu_array::get<0>(const_referenced) = 7;
    EXPECT_EQ(target, 7);
}

TEST(Tuple, ConversionAndAssignment)
{
    const auto source = gpu_array::tuple<int, int>(3, 4);
    auto converted = gpu_array::tuple<double, long>(source);
    EXPECT_EQ(gpu_array::get<0>(converted), 3.0);
    EXPECT_EQ(gpu_array::get<1>(converted), 4L);

    auto assigned = gpu_array::tuple<double, long>(0.0, 0L);
    assigned = source;
    EXPECT_EQ(gpu_array::get<0>(assigned), 3.0);
    EXPECT_EQ(gpu_array::get<1>(assigned), 4L);

    auto move_source = gpu_array::tuple<move_observer>(move_observer{42});
    auto move_constructed = gpu_array::tuple<move_observer>(std::move(move_source));
    EXPECT_EQ(gpu_array::get<0>(move_constructed).value, 42);
    EXPECT_TRUE(gpu_array::get<0>(move_source).moved_from);

    auto move_assigned = gpu_array::tuple<move_observer>(move_observer{0});
    move_assigned = std::move(move_constructed);
    EXPECT_EQ(gpu_array::get<0>(move_assigned).value, 42);
    EXPECT_TRUE(gpu_array::get<0>(move_constructed).moved_from);

    auto nested = gpu_array::tuple<gpu_array::tuple<int, double>, float>(gpu_array::tuple<int, double>(5, 6.5), 7.5F);
    EXPECT_EQ(gpu_array::get<0>(gpu_array::get<0>(nested)), 5);
    EXPECT_EQ(gpu_array::get<1>(gpu_array::get<0>(nested)), 6.5);
    EXPECT_FLOAT_EQ(gpu_array::get<1>(nested), 7.5F);
}

TEST(Tuple, Equality)
{
    EXPECT_EQ((gpu_array::tuple<int, double>(1, 2.5)), (gpu_array::tuple<int, double>(1, 2.5)));
    EXPECT_EQ((gpu_array::tuple<int, float>(1, 2.5F)), (gpu_array::tuple<long, double>(1L, 2.5)));
    EXPECT_NE((gpu_array::tuple<int, double>(1, 2.5)), (gpu_array::tuple<int, double>(1, 3.5)));
    EXPECT_EQ((gpu_array::tuple<>()), (gpu_array::tuple<>()));
}

TEST(Tuple, Apply)
{
    auto value = gpu_array::tuple<int, double, char>(2, 3.5, 'x');
    const auto result = gpu_array::apply(weighted_char_sum{}, value);
    EXPECT_DOUBLE_EQ(result, 15.5);

    gpu_array::apply(mutate_tuple_values{}, value);
    EXPECT_EQ(gpu_array::get<0>(value), 3);
    EXPECT_EQ(gpu_array::get<1>(value), 4.5);
    EXPECT_EQ(gpu_array::get<2>(value), 'y');

    const auto const_value = gpu_array::tuple<int, double>(4, 5.5);
    static_assert(std::same_as<decltype(gpu_array::apply(sum_const_values{}, const_value)), double>);
    EXPECT_DOUBLE_EQ((gpu_array::apply(sum_const_values{}, const_value)), 9.5);

    bool called = false;
    gpu_array::apply(call_marker{&called}, const_value);
    EXPECT_TRUE(called);

    auto move_value = gpu_array::tuple<move_observer>(move_observer{9});
    auto moved = gpu_array::apply(move_observer_value{}, std::move(move_value));
    EXPECT_EQ(moved.value, 9);
    EXPECT_TRUE(gpu_array::get<0>(move_value).moved_from);
}

TEST(Tuple, CommonTypeAndCommonReference)
{
    using common_type = std::common_type_t<gpu_array::tuple<int, float>, gpu_array::tuple<long, double>>;
    static_assert(std::same_as<common_type, gpu_array::tuple<long, double>>);

    using common_reference =
        std::common_reference_t<gpu_array::tuple<int&, float&>, gpu_array::tuple<const int&, const float&>>;
    static_assert(std::same_as<common_reference, gpu_array::tuple<const int&, const float&>>);

    using custom_common_type = std::common_type_t<custom_tuple<int, float, short>, custom_tuple<long, double, int>>;
    static_assert(std::same_as<custom_common_type, custom_tuple<long, double, int>>);

    using custom_common_reference =
        std::common_reference_t<custom_tuple<int&, float&, short&>, custom_tuple<const int&, const float&, const int&>>;
    static_assert(std::same_as<custom_common_reference, custom_tuple<const int&, const float&, int>>);

    SUCCEED();
}

TEST(Tuple, ReadmeStyleDerivedTuple)
{
    auto value = custom_tuple<int, float, double>(1, 2.5F, 3.5);
    EXPECT_EQ(value.get_a(), 1);
    EXPECT_FLOAT_EQ(value.get_b(), 2.5F);
    EXPECT_EQ(value.get_c(), 3.5);

    value.get_a() = 4;
    value.get_b() = 5.5F;
    value.get_c() = 6.5;
    EXPECT_EQ(gpu_array::get<0>(value), 4);
    EXPECT_FLOAT_EQ(gpu_array::get<1>(value), 5.5F);
    EXPECT_EQ(gpu_array::get<2>(value), 6.5);

    const custom_tuple<int, float, double> const_value = value;
    EXPECT_EQ(const_value.get_a(), 4);
    EXPECT_FLOAT_EQ(const_value.get_b(), 5.5F);
    EXPECT_EQ(const_value.get_c(), 6.5);
}

TEST(Tuple, UnqualifiedGetPattern)
{
    static_assert(has_unqualified_get<gpu_array::tuple<int, double>>());

    auto value = gpu_array::tuple<int, double>(8, 1.5);
    EXPECT_EQ(first_two_sum(value), 9.5);

    auto custom_value = custom_tuple<int, double, float>(2, 3.5, 4.5F);
    EXPECT_EQ(first_two_sum(custom_value), 5.5);
}

// NOLINTEND
