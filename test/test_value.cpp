#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

// NOLINTBEGIN
namespace
{
    using device_value = gpu_array::value<int>;
    using managed_int_value = gpu_array::managed_value<int>;

    template <typename T>
    concept valid_device_value = requires { sizeof(gpu_array::value<T>); };

    struct non_trivial_value
    {
        int value = 0;

        non_trivial_value() = default;
        explicit non_trivial_value(int value_) : value(value_) {}
        non_trivial_value(const non_trivial_value&) = default;
        non_trivial_value& operator=(const non_trivial_value&) = default;
        ~non_trivial_value() {}
    };

    struct value_record
    {
        int count = 0;
        float weight = 0.0F;
        double score = 0.0;

        value_record() = default;
        __host__ __device__ value_record(int count_, float weight_, double score_)
            : count(count_), weight(weight_), score(score_)
        {
        }

        friend bool operator==(const value_record&, const value_record&) = default;
    };

    struct counted_value
    {
        inline static int live_count = 0;
        inline static int copy_count = 0;
        inline static int move_count = 0;

        int value = 0;

        static void reset()
        {
            live_count = 0;
            copy_count = 0;
            move_count = 0;
        }

        counted_value() { ++live_count; }
        explicit counted_value(int value_) : value(value_) { ++live_count; }
        counted_value(const counted_value& other) : value(other.value)
        {
            ++live_count;
            ++copy_count;
        }
        counted_value(counted_value&& other) noexcept : value(other.value)
        {
            ++live_count;
            ++move_count;
        }
        counted_value& operator=(const counted_value&) = default;
        ~counted_value() { --live_count; }
    };

    struct recursive_memory_ops_probe
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

    template <typename Value>
    void expect_empty(const Value& value)
    {
        EXPECT_EQ(value.get(), nullptr);
        EXPECT_EQ(value.use_count(), 0U);
        EXPECT_FALSE(static_cast<bool>(value));
    }

    int current_device_id()
    {
        auto device_id = 0;
        GPU_CHECK_ERROR(gpu_array::api::gpuGetDevice(&device_id));
        return device_id;
    }
}  // namespace

TEST(ValueTypes, AliasesAndConstraints)
{
    static_assert(std::same_as<device_value::element_type, int>);
    static_assert(std::same_as<managed_int_value::element_type, int>);

    static_assert(valid_device_value<int>);
    static_assert(valid_device_value<value_record>);
    static_assert(!valid_device_value<non_trivial_value>);
    static_assert(requires { sizeof(gpu_array::managed_value<non_trivial_value>); });

    SUCCEED();
}

TEST(ValueCommon, EmptyAndReset)
{
    auto device = device_value();
    expect_empty(device);

    auto managed = managed_int_value();
    expect_empty(managed);

    device = device_value(17);
    managed = managed_int_value(23);
    ASSERT_TRUE(static_cast<bool>(device));
    ASSERT_TRUE(static_cast<bool>(managed));

    device.reset();
    managed.reset();
    expect_empty(device);
    expect_empty(managed);
}

TEST(ValueCommon, ConstructionAndHostRead)
{
    auto device = device_value(31);
    EXPECT_NE(device.get(), nullptr);
    EXPECT_EQ(device.use_count(), 1U);
    EXPECT_TRUE(static_cast<bool>(device));
    EXPECT_EQ(*device, 31);

    auto managed = managed_int_value(37);
    EXPECT_NE(managed.get(), nullptr);
    EXPECT_EQ(managed.use_count(), 1U);
    EXPECT_TRUE(static_cast<bool>(managed));
    EXPECT_EQ(*managed, 37);
}

TEST(ValueCommon, DefaultInitAllocates)
{
    auto device = device_value(gpu_array::default_init);
    EXPECT_NE(device.get(), nullptr);
    EXPECT_EQ(device.use_count(), 1U);
    EXPECT_TRUE(static_cast<bool>(device));

    auto managed = managed_int_value(gpu_array::default_init);
    EXPECT_NE(managed.get(), nullptr);
    EXPECT_EQ(managed.use_count(), 1U);
    EXPECT_TRUE(static_cast<bool>(managed));
}

TEST(Value, ConstructsRecordAndUsesHostProxy)
{
    const auto expected = value_record{7, 1.5F, 2.5};
    auto from_value = gpu_array::value<value_record>(expected);
    EXPECT_EQ(*from_value, expected);
    EXPECT_EQ(from_value->count, 7);
    EXPECT_FLOAT_EQ(from_value->weight, 1.5F);
    EXPECT_DOUBLE_EQ(from_value->score, 2.5);

    auto from_args = gpu_array::value<value_record>(11, 3.5F, 4.5);
    EXPECT_EQ(*from_args, (value_record{11, 3.5F, 4.5}));
    EXPECT_EQ(from_args->count, 11);
}

TEST(Value, RawPointerWrap)
{
    auto* device_pointer = static_cast<int*>(nullptr);
    constexpr auto initial_value = 101;
    GPU_CHECK_ERROR(gpu_array::api::gpuMalloc(reinterpret_cast<void**>(&device_pointer), sizeof(int)));
    GPU_CHECK_ERROR(gpu_array::api::gpuMemcpy(device_pointer, &initial_value, sizeof(int), gpuMemcpyHostToDevice));

    auto wrapped = device_value(device_pointer);
    EXPECT_EQ(wrapped.use_count(), 1U);
    EXPECT_EQ(wrapped.get(), device_pointer);
    EXPECT_EQ(*wrapped, initial_value);

    wrapped.reset();
    expect_empty(wrapped);
}

TEST(Value, RejectsManagedPointer)
{
    auto* managed_pointer = static_cast<int*>(nullptr);
    GPU_CHECK_ERROR(gpu_array::api::gpuMallocManaged(reinterpret_cast<void**>(&managed_pointer), sizeof(int)));
    *managed_pointer = 5;

    EXPECT_THROW(static_cast<void>(device_value(managed_pointer)), std::runtime_error);
    GPU_CHECK_ERROR(gpu_array::api::gpuFree(managed_pointer));
}

TEST(ManagedValue, HostAccessAndMutation)
{
    auto value = gpu_array::managed_value<value_record>(value_record{2, 4.5F, 8.5});

    EXPECT_EQ((*value).count, 2);
    value->count += 10;
    (*value).weight += 0.25F;
    value->score += 0.5;

    EXPECT_EQ((*value).count, 12);
    EXPECT_FLOAT_EQ(value->weight, 4.75F);
    EXPECT_DOUBLE_EQ(value->score, 9.0);
}

TEST(ManagedValue, NonTrivialConstruction)
{
    counted_value::reset();
    {
        auto default_value = gpu_array::managed_value<counted_value>();
        expect_empty(default_value);
        EXPECT_EQ(counted_value::live_count, 0);
    }

    counted_value::reset();
    {
        auto value = gpu_array::managed_value<counted_value>(gpu_array::default_init);
        EXPECT_NE(value.get(), nullptr);
        EXPECT_EQ(counted_value::live_count, 1);
    }
    EXPECT_EQ(counted_value::live_count, 0);

    counted_value::reset();
    {
        const auto seed = counted_value(41);
        {
            auto value = gpu_array::managed_value<counted_value>(seed);
            EXPECT_EQ(value->value, 41);
            EXPECT_EQ(counted_value::live_count, 2);
            EXPECT_EQ(counted_value::copy_count, 1);
        }
        EXPECT_EQ(counted_value::live_count, 1);
    }
    EXPECT_EQ(counted_value::live_count, 0);

    counted_value::reset();
    {
        auto value = gpu_array::managed_value<counted_value>(counted_value(43));
        EXPECT_EQ(value->value, 43);
        EXPECT_EQ(counted_value::live_count, 1);
        EXPECT_EQ(counted_value::move_count, 1);
    }
    EXPECT_EQ(counted_value::live_count, 0);

    counted_value::reset();
    {
        auto value = gpu_array::managed_value<counted_value>(47);
        EXPECT_EQ(value->value, 47);
        EXPECT_EQ(counted_value::live_count, 1);
    }
    EXPECT_EQ(counted_value::live_count, 0);
}

TEST(ManagedValue, RawPointerWrap)
{
    auto* managed_pointer = static_cast<int*>(nullptr);
    GPU_CHECK_ERROR(gpu_array::api::gpuMallocManaged(reinterpret_cast<void**>(&managed_pointer), sizeof(int)));
    *managed_pointer = 109;

    auto wrapped = managed_int_value(managed_pointer);
    EXPECT_EQ(wrapped.use_count(), 1U);
    EXPECT_EQ(wrapped.get(), managed_pointer);
    EXPECT_EQ(*wrapped, 109);

    *wrapped = 113;
    EXPECT_EQ(*managed_pointer, 113);

    wrapped.reset();
    expect_empty(wrapped);
}

TEST(ManagedValue, RejectsDevicePointer)
{
    auto* device_pointer = static_cast<int*>(nullptr);
    GPU_CHECK_ERROR(gpu_array::api::gpuMalloc(reinterpret_cast<void**>(&device_pointer), sizeof(int)));

    EXPECT_THROW(static_cast<void>(managed_int_value(device_pointer)), std::runtime_error);
    GPU_CHECK_ERROR(gpu_array::api::gpuFree(device_pointer));
}

TEST(ManagedValue, MemoryManagementApi)
{
    auto value = managed_int_value(3);
    const auto device_id = current_device_id();

    value.prefetch(device_id);
    GPU_CHECK_ERROR(gpu_array::api::gpuDeviceSynchronize());

    auto stream = gpu_array::api::gpuStream_t{};
    GPU_CHECK_ERROR(gpu_array::api::gpuStreamCreate(&stream));
    value.prefetch(stream);
    GPU_CHECK_ERROR(gpu_array::api::gpuStreamSynchronize(stream));
    GPU_CHECK_ERROR(gpu_array::api::gpuStreamDestroy(stream));

    value.mem_advise(gpu_array::api::gpuMemoryAdvise::SetReadMostly, device_id);
    value.mem_advise(gpu_array::api::gpuMemoryAdvise::UnsetReadMostly);

    value.prefetch_to_cpu();
    GPU_CHECK_ERROR(gpu_array::api::gpuDeviceSynchronize());

    value.mem_advise_to_cpu(gpu_array::api::gpuMemoryAdvise::SetPreferredLocation);
    value.mem_advise_to_cpu(gpu_array::api::gpuMemoryAdvise::UnsetPreferredLocation);

    auto empty = managed_int_value();
    empty.prefetch_to_cpu();
    empty.mem_advise_to_cpu(gpu_array::api::gpuMemoryAdvise::SetPreferredLocation);
}

TEST(ManagedValue, RecursiveMemoryManagement)
{
    auto value = gpu_array::managed_value<recursive_memory_ops_probe>(gpu_array::default_init);
    const auto device_id = current_device_id();

    value.prefetch(device_id, 0, false);
    EXPECT_EQ(value->prefetch_count, 0);

    value.prefetch(device_id, 0, true);
    EXPECT_EQ(value->prefetch_count, 1);
    EXPECT_EQ(value->prefetch_device_id, device_id);

    value.mem_advise(gpu_array::api::gpuMemoryAdvise::SetReadMostly, device_id, false);
    EXPECT_EQ(value->mem_advise_count, 0);

    value.mem_advise(gpu_array::api::gpuMemoryAdvise::SetReadMostly, device_id, true);
    EXPECT_EQ(value->mem_advise_count, 1);
    EXPECT_EQ(value->mem_advise_device_id, device_id);
    EXPECT_EQ(value->mem_advise_value, gpu_array::api::gpuMemoryAdvise::SetReadMostly);
}

// NOLINTEND
