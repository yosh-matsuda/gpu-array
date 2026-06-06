#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <list>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// NOLINTBEGIN
namespace
{
    using device_array = gpu_array::array<int, std::uint32_t>;
    using managed_int_array = gpu_array::managed_array<int, std::uint32_t>;

    template <typename T>
    concept valid_device_array_value = requires { sizeof(gpu_array::array<T>); };

    template <typename Array>
    constexpr void check_array_range_concepts()
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        static_assert(std::ranges::range<Array>);
#if defined(GPU_DEVICE_COMPILE)
        static_assert(std::ranges::borrowed_range<Array>);
#else
        static_assert(!std::ranges::borrowed_range<Array>);
#endif
        static_assert(std::ranges::view<Array>);
        static_assert(std::ranges::output_range<Array, typename Array::value_type>);
        static_assert(std::ranges::input_range<Array>);
        static_assert(std::ranges::forward_range<Array>);
        static_assert(std::ranges::bidirectional_range<Array>);
        static_assert(std::ranges::random_access_range<Array>);
        static_assert(std::ranges::sized_range<Array>);
        static_assert(std::ranges::contiguous_range<Array>);
        static_assert(std::ranges::common_range<Array>);
        static_assert(std::ranges::viewable_range<Array>);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    }

    template <typename Array>
    std::vector<typename Array::value_type> to_vector(const Array& value)
    {
        return value.template to<std::vector>();
    }

    template <typename Array>
    void expect_public_empty(const Array& value)
    {
        EXPECT_EQ(value.size(), 0U);
        EXPECT_TRUE(value.empty());
        EXPECT_EQ(value.data(), nullptr);
        EXPECT_FALSE(static_cast<bool>(value));
        EXPECT_TRUE(to_vector(value).empty());
    }

    template <typename Array>
    void expect_round_trip(const Array& value, const std::vector<typename Array::value_type>& expected)
    {
        EXPECT_EQ(value.size(), expected.size());
        EXPECT_EQ(to_vector(value), expected);
    }

    template <typename Array>
    void check_common_empty()
    {
        auto default_value = Array();
        expect_public_empty(default_value);

        auto zero_size = Array(0);
        expect_public_empty(zero_size);

        auto zero_fill = Array(0, typename Array::value_type{7});
        expect_public_empty(zero_fill);

        auto zero_default_init = Array(0, gpu_array::default_init);
        expect_public_empty(zero_default_init);
    }

    template <typename Array>
    void check_common_construction()
    {
        auto sized = Array(4);
        expect_round_trip(sized, {0, 0, 0, 0});

        auto default_initialized = Array(4, gpu_array::default_init);
        EXPECT_EQ(default_initialized.size(), 4U);
        EXPECT_FALSE(default_initialized.empty());
        EXPECT_NE(default_initialized.data(), nullptr);
        EXPECT_TRUE(static_cast<bool>(default_initialized));

        auto filled = Array(4, 7);
        expect_round_trip(filled, {7, 7, 7, 7});

        const auto vector_source = std::vector<int>{1, 2, 3, 4};
        auto from_vector = Array(vector_source);
        expect_round_trip(from_vector, vector_source);

        const auto array_source = std::array<int, 3>{5, 6, 7};
        auto from_array = Array(array_source);
        expect_round_trip(from_array, {5, 6, 7});

        const auto list_source = std::list<int>{8, 9, 10};
        auto from_list = Array(list_source);
        expect_round_trip(from_list, {8, 9, 10});

        auto from_initializer_list = Array{11, 12, 13};
        expect_round_trip(from_initializer_list, {11, 12, 13});
    }

    template <typename Array>
    void check_common_host_conversions()
    {
        const auto value = Array(std::vector<int>{3, 1, 4});

        EXPECT_EQ(value.template to<std::vector>(), (std::vector<int>{3, 1, 4}));
        EXPECT_EQ(value.template to<std::list>(), (std::list<int>{3, 1, 4}));
        EXPECT_EQ((value.template to<std::array<int, 3>>()), (std::array<int, 3>{3, 1, 4}));
        EXPECT_THROW((value.template to<std::array<int, 2>>()), std::runtime_error);

        const auto explicit_vector = static_cast<std::vector<int>>(value);
        EXPECT_EQ(explicit_vector, (std::vector<int>{3, 1, 4}));
    }

    template <typename Array>
    void check_common_value_conversions()
    {
        const auto value = Array(std::vector<int>{2, 4, 6});

        const auto as_long = value.template to<std::vector<long>>();
        EXPECT_EQ(as_long, (std::vector<long>{2L, 4L, 6L}));

        const auto as_double = value.template to<std::vector<double>>();
        EXPECT_EQ(as_double, (std::vector<double>{2.0, 4.0, 6.0}));
    }

    struct non_trivial_value
    {
        int value = 0;

        non_trivial_value() = default;
        explicit non_trivial_value(int value_) : value(value_) {}
        non_trivial_value(const non_trivial_value&) = default;
        non_trivial_value& operator=(const non_trivial_value&) = default;
        ~non_trivial_value() {}
    };

    struct counted_value
    {
        inline static int live_count = 0;
        inline static int copy_count = 0;

        int value = 0;

        static void reset()
        {
            live_count = 0;
            copy_count = 0;
        }

        counted_value() { ++live_count; }
        explicit counted_value(int value_) : value(value_) { ++live_count; }
        counted_value(const counted_value& other) : value(other.value)
        {
            ++live_count;
            ++copy_count;
        }
        counted_value(counted_value&& other) noexcept : value(other.value) { ++live_count; }
        ~counted_value() { --live_count; }
    };

    struct custom_record
    {
        int count = 0;
        float weight = 0.0F;
        double score = 0.0;

        friend bool operator==(const custom_record&, const custom_record&) = default;
    };

    struct recursive_memory_ops_element
    {
        mutable int prefetch_count = 0;
        mutable int prefetch_device_id = gpuInvalidDeviceId;
        mutable int mem_advise_count = 0;
        mutable int mem_advise_device_id = gpuInvalidDeviceId;
        mutable gpu_array::api::gpuMemoryAdvise mem_advise_value = gpu_array::api::gpuMemoryAdvise::UnsetReadMostly;

        [[maybe_unused]] void prefetch(int device_id, gpu_array::api::gpuStream_t) const
        {
            ++prefetch_count;
            prefetch_device_id = device_id;
        }

        [[maybe_unused]] void mem_advise(gpu_array::api::gpuMemoryAdvise advise, int device_id) const
        {
            ++mem_advise_count;
            mem_advise_device_id = device_id;
            mem_advise_value = advise;
        }
    };
}  // namespace

TEST(ArrayTypes, AliasesAndConstraints)
{
    static_assert(std::same_as<device_array::value_type, int>);
    static_assert(std::same_as<device_array::size_type, std::uint32_t>);
    static_assert(std::same_as<device_array::pointer, int*>);
    static_assert(std::same_as<device_array::const_pointer, const int*>);

    static_assert(std::same_as<managed_int_array::value_type, int>);
    static_assert(std::same_as<managed_int_array::size_type, std::uint32_t>);

    check_array_range_concepts<device_array>();
    check_array_range_concepts<managed_int_array>();

    static_assert(valid_device_array_value<int>);
    static_assert(!valid_device_array_value<non_trivial_value>);
    static_assert(requires { sizeof(gpu_array::managed_array<non_trivial_value>); });

    SUCCEED();
}

TEST(ArrayCommon, Empty)
{
    check_common_empty<device_array>();
    check_common_empty<managed_int_array>();
}

TEST(ArrayCommon, Construction)
{
    check_common_construction<device_array>();
    check_common_construction<managed_int_array>();

    const auto source = std::vector<int>{1, 2, 3};
    auto deduced_device = gpu_array::array(source);
    auto deduced_managed = gpu_array::managed_array(source);
    static_assert(std::same_as<typename decltype(deduced_device)::value_type, int>);
    static_assert(std::same_as<typename decltype(deduced_managed)::value_type, int>);
    EXPECT_EQ(deduced_device.template to<std::vector>(), source);
    EXPECT_EQ(deduced_managed.template to<std::vector>(), source);
}

TEST(ArrayCommon, HostConversions)
{
    check_common_host_conversions<device_array>();
    check_common_host_conversions<managed_int_array>();
}

TEST(ArrayCommon, ValueConversions)
{
    check_common_value_conversions<device_array>();
    check_common_value_conversions<managed_int_array>();
}

TEST(ArrayCommon, CrossMemoryConversions)
{
    auto device_value = device_array(std::vector<int>{1, 3, 5});
    auto managed_from_device = device_value.to<managed_int_array>();
    expect_round_trip(managed_from_device, {1, 3, 5});

    auto device_long = device_value.to<gpu_array::array<long, std::uint32_t>>();
    EXPECT_EQ(device_long.template to<std::vector>(), (std::vector<long>{1L, 3L, 5L}));

    auto managed_value = managed_int_array(std::vector<int>{2, 4, 6});
    auto device_from_managed = managed_value.to<device_array>();
    expect_round_trip(device_from_managed, {2, 4, 6});

    auto managed_double = managed_value.to<gpu_array::managed_array<double, std::uint32_t>>();
    EXPECT_EQ(managed_double.template to<std::vector>(), (std::vector<double>{2.0, 4.0, 6.0}));
}

TEST(ManagedArray, HostAccess)
{
    auto value = managed_int_array{1, 2, 3, 4};

    value[0] = 10;
    value.front() += 1;
    value.back() = 40;
    EXPECT_EQ(value[0], 11);
    EXPECT_EQ(value[3], 40);

    for (auto& element : value) element += 1;
    EXPECT_EQ(value.template to<std::vector>(), (std::vector<int>{12, 3, 4, 41}));

    const auto& const_value = value;
    EXPECT_EQ(*const_value.begin(), 12);
    EXPECT_EQ(*(const_value.end() - 1), 41);
    EXPECT_EQ(*const_value.cbegin(), 12);
    EXPECT_EQ(*const_value.crbegin(), 41);

    auto reverse_values = std::vector<int>();
    for (auto it = value.rbegin(); it != value.rend(); ++it) reverse_values.push_back(*it);
    EXPECT_EQ(reverse_values, (std::vector<int>{41, 4, 3, 12}));
}

TEST(ManagedArray, NonTrivialConstruction)
{
    counted_value::reset();
    {
        auto values = gpu_array::managed_array<counted_value, std::uint32_t>(3);
        EXPECT_EQ(values.size(), 3U);
        EXPECT_EQ(counted_value::live_count, 3);
    }
    EXPECT_EQ(counted_value::live_count, 0);

    counted_value::reset();
    {
        auto seed = counted_value(9);
        {
            auto values = gpu_array::managed_array<counted_value, std::uint32_t>(2, seed);
            EXPECT_EQ(values.size(), 2U);
            EXPECT_EQ(values[0].value, 9);
            EXPECT_EQ(values[1].value, 9);
            EXPECT_EQ(counted_value::live_count, 3);
            EXPECT_EQ(counted_value::copy_count, 2);
        }
        EXPECT_EQ(counted_value::live_count, 1);
    }
    EXPECT_EQ(counted_value::live_count, 0);

    counted_value::reset();
    {
        const auto source = std::array<counted_value, 2>{counted_value(4), counted_value(5)};
        {
            auto values = gpu_array::managed_array<counted_value, std::uint32_t>(source);
            EXPECT_EQ(values.size(), 2U);
            EXPECT_EQ(values[0].value, 4);
            EXPECT_EQ(values[1].value, 5);
            EXPECT_EQ(counted_value::live_count, 4);
        }
        EXPECT_EQ(counted_value::live_count, 2);
    }
    EXPECT_EQ(counted_value::live_count, 0);
}

TEST(ManagedArray, NestedRoundTrip)
{
    const auto source = std::vector<std::vector<int>>{{1, 2, 3}, {4}, {}, {5, 6}};

    auto value = gpu_array::managed_array(source);
    static_assert(std::same_as<typename decltype(value)::value_type, gpu_array::managed_array<int>>);

    ASSERT_EQ(value.size(), source.size());
    for (auto i = std::size_t{0}; i < source.size(); ++i)
    {
        EXPECT_EQ(value[i].template to<std::vector>(), source[i]);
    }

    auto round_trip = value.template to<std::vector>();
    EXPECT_EQ(round_trip, source);
}

TEST(ManagedArray, CustomStructRoundTrip)
{
    const auto source = std::vector<custom_record>{{1, 1.5F, 2.5}, {3, 4.5F, 5.5}, {6, 7.5F, 8.5}};

    auto value = gpu_array::managed_array<custom_record, std::uint32_t>(source);

    ASSERT_EQ(value.size(), source.size());
    value[1].count += 10;
    value[1].weight += 0.25F;
    value[1].score += 0.5;

    auto expected = source;
    expected[1] = custom_record{13, 4.75F, 6.0};
    EXPECT_EQ(value.template to<std::vector>(), expected);
}

TEST(ManagedArray, RecursiveMemoryManagement)
{
    auto values = gpu_array::managed_array<recursive_memory_ops_element, std::uint32_t>(3, gpu_array::default_init);
    auto device_id = 0;
    GPU_CHECK_ERROR(gpu_array::api::gpuGetDevice(&device_id));

    values.prefetch(device_id, 0, false);
    EXPECT_EQ(values[0].prefetch_count, 0);
    EXPECT_EQ(values[1].prefetch_count, 0);
    EXPECT_EQ(values[2].prefetch_count, 0);

    values.prefetch(1, 2, device_id, 0, true);
    EXPECT_EQ(values[0].prefetch_count, 0);
    EXPECT_EQ(values[1].prefetch_count, 1);
    EXPECT_EQ(values[1].prefetch_device_id, device_id);
    EXPECT_EQ(values[2].prefetch_count, 1);
    EXPECT_EQ(values[2].prefetch_device_id, device_id);

    values.mem_advise(gpu_array::api::gpuMemoryAdvise::SetReadMostly, device_id, false);
    EXPECT_EQ(values[0].mem_advise_count, 0);
    EXPECT_EQ(values[1].mem_advise_count, 0);
    EXPECT_EQ(values[2].mem_advise_count, 0);

    values.mem_advise(0, 2, gpu_array::api::gpuMemoryAdvise::SetReadMostly, device_id, true);
    EXPECT_EQ(values[0].mem_advise_count, 1);
    EXPECT_EQ(values[0].mem_advise_device_id, device_id);
    EXPECT_EQ(values[0].mem_advise_value, gpu_array::api::gpuMemoryAdvise::SetReadMostly);
    EXPECT_EQ(values[1].mem_advise_count, 1);
    EXPECT_EQ(values[1].mem_advise_device_id, device_id);
    EXPECT_EQ(values[1].mem_advise_value, gpu_array::api::gpuMemoryAdvise::SetReadMostly);
    EXPECT_EQ(values[2].mem_advise_count, 0);
}

TEST(ArrayPointerWrap, RawPointers)
{
    const auto source = std::vector<int>{7, 8, 9};

    auto* device_pointer = static_cast<int*>(nullptr);
    GPU_CHECK_ERROR(gpu_array::api::gpuMalloc(reinterpret_cast<void**>(&device_pointer), sizeof(int) * source.size()));
    GPU_CHECK_ERROR(
        gpu_array::api::gpuMemcpy(device_pointer, source.data(), sizeof(int) * source.size(), gpuMemcpyHostToDevice));
    {
        auto wrapped = device_array(device_pointer, source.size());
        expect_round_trip(wrapped, source);
        wrapped.reset();
        expect_public_empty(wrapped);
    }

    auto* managed_pointer = static_cast<int*>(nullptr);
    GPU_CHECK_ERROR(
        gpu_array::api::gpuMallocManaged(reinterpret_cast<void**>(&managed_pointer), sizeof(int) * source.size()));
    std::copy(source.begin(), source.end(), managed_pointer);
    {
        auto wrapped = managed_int_array(managed_pointer, source.size());
        EXPECT_EQ(wrapped.template to<std::vector>(), source);
        EXPECT_EQ(wrapped[1], 8);
        wrapped.reset();
        expect_public_empty(wrapped);
    }
}

// NOLINTEND
