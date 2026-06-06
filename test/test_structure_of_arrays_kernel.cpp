#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <vector>

// NOLINTBEGIN
namespace
{
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

    void synchronize() { GPU_CHECK_ERROR(gpu_array::api::gpuDeviceSynchronize()); }

    template <typename Records>
    __global__ void update_record_interface_kernel(Records records)
    {
        using size_type = typename Records::size_type;
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto index = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); index < records.size();
             index += stride)
        {
            auto record = records[index];
            record.get_a() += 10;
            record.get_b() *= 2.0F;
            record.get_c() += 0.25;
        }
    }

    template <typename Records>
    __global__ void update_member_arrays_kernel(Records records)
    {
        using size_type = typename Records::size_type;
        auto* a = records.template data<0>();
        auto* b = records.template data<1>();
        auto* c = records.template data<2>();
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto index = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); index < records.size();
             index += stride)
        {
            a[index] += 1;
            b[index] += 2.0F;
            c[index] += 3.0;
        }
    }

    template <std::ranges::input_range Records>
    __global__ void update_grid_thread_stride_view_kernel(Records records)
    {
        for (auto&& value : records | gpu_array::views::grid_thread_stride)
        {
            value.get_a() = value.get_a() * 2 + 1;
            value.get_b() += 0.5F;
            value.get_c() += 1.0;
        }
    }

    template <std::ranges::input_range Records>
    __global__ void enumerate_stride_view_kernel(Records records)
    {
        for (auto&& [index, value] : records | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride)
        {
            value.get_a() += static_cast<int>(index) * 10;
            value.get_b() += static_cast<float>(index);
            value.get_c() += static_cast<double>(index) * 0.25;
        }
    }

    template <std::ranges::input_range Lhs, std::ranges::input_range Rhs>
    __global__ void zip_stride_view_kernel(Lhs lhs, Rhs rhs)
    {
        for (auto&& [left, right] : gpu_array::views::zip(lhs, rhs) | gpu_array::views::grid_thread_stride)
        {
            left.get_a() += right.get_a();
            left.get_b() += right.get_b();
            left.get_c() += right.get_c();
        }
    }

    template <typename Record>
    std::vector<Record> make_records()
    {
        return {Record{1, 1.5F, 2.5}, Record{2, 3.5F, 4.5}, Record{3, 5.5F, 6.5}, Record{4, 7.5F, 8.5}};
    }

    template <typename Record>
    std::vector<Record> record_interface_expected(std::vector<Record> values)
    {
        for (auto& value : values)
        {
            value.get_a() += 10;
            value.get_b() *= 2.0F;
            value.get_c() += 0.25;
        }
        return values;
    }

    template <typename Record>
    std::vector<Record> member_arrays_expected(std::vector<Record> values)
    {
        for (auto& value : values)
        {
            value.get_a() += 1;
            value.get_b() += 2.0F;
            value.get_c() += 3.0;
        }
        return values;
    }

    template <typename Record, typename Records>
    std::vector<Record> copy_member_arrays(const Records& records)
    {
        const auto size = static_cast<std::size_t>(records.size());
        auto a = std::vector<int>(size);
        auto b = std::vector<float>(size);
        auto c = std::vector<double>(size);
        GPU_CHECK_ERROR(
            gpu_array::api::gpuMemcpy(a.data(), records.template data<0>(), sizeof(int) * size, gpuMemcpyDeviceToHost));
        GPU_CHECK_ERROR(gpu_array::api::gpuMemcpy(b.data(), records.template data<1>(), sizeof(float) * size,
                                                  gpuMemcpyDeviceToHost));
        GPU_CHECK_ERROR(gpu_array::api::gpuMemcpy(c.data(), records.template data<2>(), sizeof(double) * size,
                                                  gpuMemcpyDeviceToHost));

        auto result = std::vector<Record>{};
        result.reserve(size);
        for (auto index = std::size_t{0}; index < size; ++index)
        {
            result.emplace_back(a[index], b[index], c[index]);
        }
        return result;
    }
}  // namespace

template <class... TTypes, class... UTypes>
requires requires { typename gpu_tuple_record<std::common_type_t<TTypes, UTypes>...>; }
struct std::common_type<gpu_tuple_record<TTypes...>, gpu_tuple_record<UTypes...>>
{
    using type = gpu_tuple_record<std::common_type_t<TTypes, UTypes>...>;
};

template <class... TTypes, class... UTypes, template <class> class TQual, template <class> class UQual>
requires requires { typename gpu_tuple_record<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>; }
struct std::basic_common_reference<gpu_tuple_record<TTypes...>, gpu_tuple_record<UTypes...>, TQual, UQual>
{
    using type = gpu_tuple_record<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>;
};

template <class... TTypes, class... UTypes>
requires requires { typename std_tuple_record<std::common_type_t<TTypes, UTypes>...>; }
struct std::common_type<std_tuple_record<TTypes...>, std_tuple_record<UTypes...>>
{
    using type = std_tuple_record<std::common_type_t<TTypes, UTypes>...>;
};

template <class... TTypes, class... UTypes, template <class> class TQual, template <class> class UQual>
requires requires { typename std_tuple_record<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>; }
struct std::basic_common_reference<std_tuple_record<TTypes...>, std_tuple_record<UTypes...>, TQual, UQual>
{
    using type = std_tuple_record<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>;
};

TEST(ManagedStructureOfArraysKernel, TupleRecords)
{
    {
        using record = gpu_tuple_record<int, float, double>;
        const auto source = make_records<record>();
        auto records = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(source);

        update_record_interface_kernel<<<2, 3>>>(records);
        synchronize();

        expect_records_eq(records.template to<std::vector>(), record_interface_expected(source));
    }
    {
        using record = std_tuple_record<int, float, double>;
        const auto source = make_records<record>();
        auto records = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(source);

        update_record_interface_kernel<<<2, 3>>>(records);
        synchronize();

        expect_records_eq(records.template to<std::vector>(), record_interface_expected(source));
    }
}

TEST(StructureOfArraysKernel, DataPointers)
{
    using record = gpu_tuple_record<int, float, double>;
    const auto source = make_records<record>();
    auto records = gpu_array::structure_of_arrays<record, std::uint32_t>(source);

    update_member_arrays_kernel<<<2, 3>>>(records);
    synchronize();

    expect_records_eq(copy_member_arrays<record>(records), member_arrays_expected(source));
}

TEST(ManagedStructureOfArraysKernel, GridThreadStrideView)
{
    using record = gpu_tuple_record<int, float, double>;
    const auto source = make_records<record>();
    auto records = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(source);

    update_grid_thread_stride_view_kernel<<<2, 3>>>(records);
    synchronize();

    expect_records_eq(
        records.template to<std::vector>(),
        std::vector<record>{record{3, 2.0F, 3.5}, record{5, 4.0F, 5.5}, record{7, 6.0F, 7.5}, record{9, 8.0F, 9.5}});
}

TEST(ManagedStructureOfArraysKernel, EnumerateStrideView)
{
    using record = gpu_tuple_record<int, float, double>;
    const auto source = make_records<record>();
    auto records = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(source);

    enumerate_stride_view_kernel<<<2, 3>>>(records);
    synchronize();

    expect_records_eq(records.template to<std::vector>(),
                      std::vector<record>{record{1, 1.5F, 2.5}, record{12, 4.5F, 4.75}, record{23, 7.5F, 7.0},
                                          record{34, 10.5F, 9.25}});
}

TEST(ManagedStructureOfArraysKernel, ZipStrideView)
{
    using record = gpu_tuple_record<int, float, double>;
    const auto source = make_records<record>();
    auto lhs = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(source);
    auto rhs = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(
        std::vector<record>{record{10, 10.0F, 10.0}, record{20, 20.0F, 20.0}, record{30, 30.0F, 30.0}});

    zip_stride_view_kernel<<<2, 3>>>(lhs, rhs);
    synchronize();

    expect_records_eq(lhs.template to<std::vector>(),
                      std::vector<record>{record{11, 11.5F, 12.5}, record{22, 23.5F, 24.5}, record{33, 35.5F, 36.5},
                                          record{4, 7.5F, 8.5}});
}

// NOLINTEND
