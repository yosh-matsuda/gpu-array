#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

// NOLINTBEGIN
namespace
{
    template <bool Unified, typename SizeType, typename... ValueTypes>
    class base_probe : public gpu_array::detail::base<Unified, SizeType, ValueTypes...>
    {
        using base_type = gpu_array::detail::base<Unified, SizeType, ValueTypes...>;

    public:
        using base_type::empty;
        using base_type::size;
        using base_type::use_count;
        using typename base_type::size_type;

        base_probe() = default;
        explicit base_probe(std::size_t count) : base_type(count) {}
        base_probe(const base_probe&) = default;
        base_probe(base_probe&&) noexcept = default;

        template <std::size_t Index>
        using element_type = std::tuple_element_t<Index, gpu_array::tuple<ValueTypes...>>;

        template <std::size_t Index>
        __host__ __device__ element_type<Index>* data() const noexcept
        {
            return gpu_array::get<Index>(base_type::data_);
        }

        static base_probe allocate_owned(std::size_t count)
        {
            auto result = base_probe(count);
            result.allocate_all();
            return result;
        }

        base_probe& copy_assign(const base_probe& other)
        {
            base_type::operator=(other);
            return *this;
        }

        base_probe& move_assign(base_probe&& other) noexcept
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        void reset() { base_type::free(); }

    private:
        void allocate_all()
        {
            if (base_type::size_ == 0) return;
            allocate_all(std::index_sequence_for<ValueTypes...>{});
        }

        template <std::size_t... Indices>
        void allocate_all(std::index_sequence<Indices...>)
        {
            (allocate_one<Indices>(), ...);
        }

        template <std::size_t Index>
        void allocate_one()
        {
            using value_type = element_type<Index>;
            void* raw_pointer = nullptr;
            if constexpr (Unified)
            {
                GPU_CHECK_ERROR(gpu_array::api::gpuMallocManaged(&raw_pointer, sizeof(value_type) * base_type::size_));
                std::ranges::uninitialized_default_construct_n(static_cast<value_type*>(raw_pointer), base_type::size_);
            }
            else
            {
                GPU_CHECK_ERROR(gpu_array::api::gpuMalloc(&raw_pointer, sizeof(value_type) * base_type::size_));
            }
            gpu_array::get<Index>(base_type::data_) = static_cast<value_type*>(raw_pointer);
        }
    };

    template <typename ValueType>
    class range_probe : public base_probe<false, std::size_t, ValueType>
    {
    public:
        using value_type = ValueType;

        ValueType* begin() const noexcept { return nullptr; }
        ValueType* end() const noexcept { return nullptr; }
    };

    struct counted_object
    {
        inline static int live_count = 0;

        counted_object() { ++live_count; }
        counted_object(const counted_object&) { ++live_count; }
        counted_object(counted_object&&) noexcept { ++live_count; }
        ~counted_object() { --live_count; }
    };

    template <typename Probe>
    void expect_empty(const Probe& value)
    {
        EXPECT_EQ(value.size(), 0U);
        EXPECT_TRUE(value.empty());
        EXPECT_EQ(value.use_count(), 0U);
        EXPECT_EQ(value.template data<0>(), nullptr);
    }
}  // namespace

TEST(DetailBase, Empty)
{
    auto default_value = base_probe<false, std::uint32_t, int>();
    expect_empty(default_value);

    auto zero_size = base_probe<false, std::uint32_t, int>(0);
    expect_empty(zero_size);

    auto copied = zero_size;
    expect_empty(copied);

    auto moved = base_probe<false, std::uint32_t, int>(std::move(zero_size));
    expect_empty(moved);
    EXPECT_EQ(zero_size.data<0>(), nullptr);
}

TEST(DetailBase, Sharing)
{
    auto owner = base_probe<false, std::uint32_t, int>::allocate_owned(4);
    const auto* owner_data = owner.data<0>();
    ASSERT_NE(owner_data, nullptr);
    EXPECT_EQ(owner.size(), 4U);
    EXPECT_EQ(owner.use_count(), 1U);

    auto copied = owner;
    EXPECT_EQ(copied.data<0>(), owner_data);
    EXPECT_EQ(owner.use_count(), 2U);
    EXPECT_EQ(copied.use_count(), 2U);

    auto assigned = base_probe<false, std::uint32_t, int>();
    assigned.copy_assign(owner);
    EXPECT_EQ(assigned.data<0>(), owner_data);
    EXPECT_EQ(owner.use_count(), 3U);
    EXPECT_EQ(copied.use_count(), 3U);
    EXPECT_EQ(assigned.use_count(), 3U);

    auto moved = base_probe<false, std::uint32_t, int>(std::move(assigned));
    EXPECT_EQ(moved.data<0>(), owner_data);
    EXPECT_EQ(assigned.size(), 0U);
    EXPECT_EQ(assigned.use_count(), 0U);
    EXPECT_EQ(assigned.data<0>(), nullptr);
    EXPECT_EQ(owner.use_count(), 3U);

    moved.reset();
    EXPECT_EQ(owner.use_count(), 2U);

    owner.reset();
    EXPECT_EQ(owner.data<0>(), nullptr);
    EXPECT_EQ(copied.use_count(), 1U);

    copied.reset();
    expect_empty(copied);
}

TEST(DetailBase, MoveAssign)
{
    auto source = base_probe<false, std::uint32_t, int>::allocate_owned(2);
    const auto* source_data = source.data<0>();
    ASSERT_NE(source_data, nullptr);

    auto target = base_probe<false, std::uint32_t, int>::allocate_owned(1);
    ASSERT_NE(target.data<0>(), nullptr);

    target.move_assign(std::move(source));
    EXPECT_EQ(target.data<0>(), source_data);
    EXPECT_EQ(target.size(), 2U);
    EXPECT_EQ(target.use_count(), 1U);
    expect_empty(source);
}

TEST(DetailBase, SelfAssign)
{
    auto owner = base_probe<false, std::uint32_t, int>::allocate_owned(3);
    const auto* owner_data = owner.data<0>();
    ASSERT_NE(owner_data, nullptr);

    owner.copy_assign(owner);
    EXPECT_EQ(owner.data<0>(), owner_data);
    EXPECT_EQ(owner.size(), 3U);
    EXPECT_EQ(owner.use_count(), 1U);

    owner.move_assign(std::move(owner));
    EXPECT_EQ(owner.data<0>(), owner_data);
    EXPECT_EQ(owner.size(), 3U);
    EXPECT_EQ(owner.use_count(), 1U);
}

TEST(DetailBase, Overflow)
{
    constexpr auto overflowing_size = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1U;
    EXPECT_THROW((base_probe<false, std::uint32_t, int>(overflowing_size)), std::runtime_error);
}

TEST(DetailBase, UnifiedFree)
{
    counted_object::live_count = 0;
    {
        auto owner = base_probe<true, std::uint32_t, counted_object>::allocate_owned(3);
        EXPECT_EQ(owner.size(), 3U);
        EXPECT_EQ(owner.use_count(), 1U);
        EXPECT_NE(owner.data<0>(), nullptr);
        EXPECT_EQ(counted_object::live_count, 3);
    }
    EXPECT_EQ(counted_object::live_count, 0);
}

TEST(DetailBase, DebugCounter)
{
#if defined(GPU_ARRAY_DEBUG)
    gpu_array::detail::gpu_memory_usage = 0U;
    {
        auto owner = base_probe<false, std::uint32_t, int, double>::allocate_owned(5);
        EXPECT_EQ(gpu_array::detail::gpu_memory_usage, (sizeof(int) + sizeof(double)) * 5U);

        auto copied = owner;
        EXPECT_EQ(gpu_array::detail::gpu_memory_usage, (sizeof(int) + sizeof(double)) * 5U);

        auto moved = base_probe<false, std::uint32_t, int, double>(std::move(copied));
        EXPECT_NE(moved.data<0>(), nullptr);
        EXPECT_NE(moved.data<1>(), nullptr);
        EXPECT_EQ(gpu_array::detail::gpu_memory_usage, (sizeof(int) + sizeof(double)) * 5U);
    }
    EXPECT_EQ(gpu_array::detail::gpu_memory_usage, 0U);
#else
    SUCCEED();
#endif
}

TEST(DetailTraits, Pointers)
{
    using unmanaged_array_probe = base_probe<false, std::uint32_t, int>;
    using managed_array_probe = base_probe<true, std::uint32_t, int>;
    using unmanaged_tuple_probe = base_probe<false, std::uint32_t, int, double>;

    static_assert(gpu_array::detail::gpu_ptr<unmanaged_array_probe>);
    static_assert(gpu_array::detail::gpu_unmanaged_ptr<unmanaged_array_probe>);
    static_assert(!gpu_array::detail::gpu_managed_ptr<unmanaged_array_probe>);
    static_assert(gpu_array::detail::gpu_array_ptr<unmanaged_array_probe>);
    static_assert(gpu_array::detail::gpu_unmanaged_array_ptr<unmanaged_array_probe>);
    static_assert(!gpu_array::detail::gpu_managed_array_ptr<unmanaged_array_probe>);

    static_assert(gpu_array::detail::gpu_ptr<managed_array_probe>);
    static_assert(gpu_array::detail::gpu_managed_ptr<managed_array_probe>);
    static_assert(!gpu_array::detail::gpu_unmanaged_ptr<managed_array_probe>);
    static_assert(gpu_array::detail::gpu_array_ptr<managed_array_probe>);
    static_assert(gpu_array::detail::gpu_managed_array_ptr<managed_array_probe>);
    static_assert(!gpu_array::detail::gpu_unmanaged_array_ptr<managed_array_probe>);

    static_assert(gpu_array::detail::gpu_ptr<unmanaged_tuple_probe>);
    static_assert(gpu_array::detail::gpu_unmanaged_ptr<unmanaged_tuple_probe>);
    static_assert(!gpu_array::detail::gpu_array_ptr<unmanaged_tuple_probe>);

    EXPECT_TRUE(gpu_array::detail::gpu_ptr<unmanaged_array_probe>);
}

TEST(DetailTraits, Ranges)
{
    static_assert(gpu_array::detail::array_convertible<std::array<int, 3>>);
    static_assert(gpu_array::detail::array_convertible<std::vector<int>>);
    static_assert(!gpu_array::detail::array_convertible<range_probe<int>>);
    static_assert(!gpu_array::detail::array_convertible_for_copy<range_probe<int>>);

    using deduced_scalar = gpu_array::detail::unified_array_deduced_t<int>;
    using deduced_range = gpu_array::detail::unified_array_deduced_t<std::array<int, 3>>;
    using deduced_nested_range = gpu_array::detail::unified_array_deduced_t<std::array<std::array<int, 2>, 3>>;
    static_assert(std::same_as<deduced_scalar, int>);
    static_assert(std::same_as<deduced_range, gpu_array::managed_array<int>>);
    static_assert(std::same_as<deduced_nested_range, gpu_array::managed_array<gpu_array::managed_array<int>>>);

    using vector_deduced = gpu_array::detail::to_range_deduced_t<range_probe<int>, std::vector>;
    using nested_vector_deduced = gpu_array::detail::to_range_deduced_t<range_probe<range_probe<int>>, std::vector>;
    static_assert(std::same_as<vector_deduced, std::vector<int>>);
    static_assert(std::same_as<nested_vector_deduced, std::vector<std::vector<int>>>);

    EXPECT_TRUE((gpu_array::detail::array_convertible<std::array<int, 3>>));
}

TEST(DetailSubrange, Bounds)
{
    auto values = std::array<int, 4>{1, 2, 3, 4};
    auto view = gpu_array::detail::subrange<int*, std::uint32_t>(values.data() + 1, 2);
    EXPECT_EQ(view.begin(), values.data() + 1);
    EXPECT_EQ(view.end(), values.data() + 3);
    EXPECT_EQ(view.size(), 2U);
    EXPECT_FALSE(view.empty());
    EXPECT_EQ(*view.begin(), 2);

    auto empty_view = gpu_array::detail::subrange<int*, std::uint32_t>(values.data(), 0);
    EXPECT_EQ(empty_view.begin(), values.data());
    EXPECT_EQ(empty_view.end(), values.data());
    EXPECT_EQ(empty_view.size(), 0U);
    EXPECT_TRUE(empty_view.empty());
}

TEST(DetailSubrange, ConceptsAndAccessors)
{
    using view_type = gpu_array::detail::subrange<int*, std::uint32_t>;
    static_assert(std::ranges::view<view_type>);
    static_assert(std::ranges::borrowed_range<view_type>);
    static_assert(std::ranges::sized_range<view_type>);
    static_assert(std::ranges::common_range<view_type>);
    static_assert(std::ranges::random_access_range<view_type>);
    static_assert(std::ranges::contiguous_range<view_type>);

    auto values = std::array<int, 4>{1, 2, 3, 4};
    auto view = view_type(values.data(), values.size());

    EXPECT_TRUE(static_cast<bool>(view));
    EXPECT_EQ(view.front(), 1);
    EXPECT_EQ(view.back(), 4);
    EXPECT_EQ(view[2], 3);
    EXPECT_EQ(std::ranges::data(view), values.data());

    view[1] = 20;
    EXPECT_EQ(values[1], 20);
}

// NOLINTEND
