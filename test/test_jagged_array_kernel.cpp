#include "gpu_array.hpp"

#include "constrained_kernel_launch.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
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

    void synchronize() { GPU_CHECK_ERROR(gpu_array::api::gpuDeviceSynchronize()); }

    auto make_jagged_ints() { return std::vector<std::vector<int>>{{1}, {2, 3}, {}, {4, 5, 6}}; }

    template <typename Jagged>
    __global__ void update_flat_range_kernel(Jagged values)
    {
        using size_type = typename Jagged::size_type;
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto index = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); index < values.size();
             index += stride)
        {
            values[index] = values[index] * 2 + static_cast<int>(index);
        }
    }

    template <typename Jagged>
    __global__ void update_rows_kernel(Jagged values)
    {
        using size_type = typename Jagged::size_type;
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto row = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); row < values.num_rows();
             row += stride)
        {
            for (auto column = size_type{0}; column < values.size(row); ++column)
            {
                values[{row, column}] += static_cast<int>(row * 100 + column);
            }
        }
    }

    template <typename Jagged>
    __global__ void update_soa_rows_kernel(Jagged records)
    {
        using size_type = typename Jagged::size_type;
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto row = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); row < records.num_rows();
             row += stride)
        {
            for (auto column = size_type{0}; column < records.size(row); ++column)
            {
                auto record_ref = records[{row, column}];
                record_ref.get_a() += static_cast<int>(row * 10 + column);
                record_ref.get_b() *= 2.0F;
                record_ref.get_c() += 0.5;
            }
        }
    }

    template <typename Jagged>
    __global__ void update_soa_flat_data_kernel(Jagged records)
    {
        using size_type = typename Jagged::size_type;
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

    template <typename Jagged>
    __global__ void update_soa_row_data_kernel(Jagged records)
    {
        using size_type = typename Jagged::size_type;
        const auto stride = static_cast<size_type>(blockDim.x * gridDim.x);
        for (auto row = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x); row < records.num_rows();
             row += stride)
        {
            auto* a = records.template data<0>(row);
            auto* b = records.template data<1>(row);
            auto* c = records.template data<2>(row);
            for (auto column = size_type{0}; column < records.size(row); ++column)
            {
                a[column] += 2;
                b[column] += 4.0F;
                c[column] += 6.0;
            }
        }
    }

    template <typename Jagged>
    __global__ void update_flat_views_kernel(Jagged values)
    {
        for (auto&& [index, value] : values | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride)
        {
            value += static_cast<int>(index) * 10;
        }
    }

    template <typename Jagged>
    __global__ void update_row_views_kernel(Jagged values)
    {
        for (auto row_index = static_cast<typename Jagged::size_type>(blockIdx.x); row_index < values.num_rows();
             row_index += static_cast<typename Jagged::size_type>(gridDim.x))
        {
            auto row = values.row(row_index);
            for (auto&& [column_index, value] :
                 row | gpu_array::views::enumerate | gpu_array::views::block_thread_stride)
            {
                value += static_cast<int>(row_index * 100 + column_index);
            }
        }
    }

    template <typename Jagged>
    __global__ void update_soa_flat_views_kernel(Jagged records)
    {
        for (auto&& [index, value] : records | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride)
        {
            value.get_a() += static_cast<int>(index) * 100;
        }
    }

    template <typename Jagged>
    __global__ void update_soa_row_views_kernel(Jagged records)
    {
        for (auto row_index = static_cast<typename Jagged::size_type>(blockIdx.x); row_index < records.num_rows();
             row_index += static_cast<typename Jagged::size_type>(gridDim.x))
        {
            auto row = records.row(row_index);
            for (auto&& [column_index, value] :
                 row | gpu_array::views::enumerate | gpu_array::views::block_thread_stride)
            {
                value.get_b() += static_cast<float>(row_index * 10 + column_index);
            }
        }
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

TEST(JaggedArrayKernel, FlatRange)
{
    const auto source = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto values = managed_int_jagged({1U, 2U, 3U, 4U}, source);

    update_flat_range_kernel<<<2, 5>>>(values);
    synchronize();

    auto expected = source;
    for (auto index = std::size_t{0}; index < expected.size(); ++index)
    {
        expected[index] = expected[index] * 2 + static_cast<int>(index);
    }
    EXPECT_EQ(values.template to<std::vector>(), expected);
}

TEST(JaggedArrayKernel, Rows)
{
    const auto source = std::vector<std::vector<int>>{{1}, {2, 3}, {}, {4, 5, 6}, {7, 8}};
    auto values = managed_int_jagged(source);

    update_rows_kernel<<<2, 4>>>(values);
    synchronize();

    auto expected = std::vector<int>{1, 102, 104, 304, 306, 308, 407, 409};
    EXPECT_EQ(values.template to<std::vector>(), expected);
}

TEST(JaggedArrayKernel, FlatViews)
{
    auto values = managed_int_jagged(make_jagged_ints());

    gpu_array_test::launch_input_range<update_flat_views_kernel<decltype(values)>>({2}, {3}, values);
    synchronize();

    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{1, 12, 23, 34, 45, 56}));
}

TEST(JaggedArrayKernel, RowViews)
{
    auto values = managed_int_jagged(make_jagged_ints());

    update_row_views_kernel<<<2, 3>>>(values);
    synchronize();

    EXPECT_EQ(values.template to<std::vector>(), (std::vector<int>{1, 102, 104, 304, 306, 308}));
}

TEST(JaggedArrayKernel, GpuTupleSoa)
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
    auto records = managed_record_jagged(nested);

    update_soa_rows_kernel<<<2, 4>>>(records);
    synchronize();
    update_soa_flat_data_kernel<<<2, 4>>>(records);
    synchronize();
    update_soa_row_data_kernel<<<2, 4>>>(records);
    synchronize();

    auto expected = std::vector<record>{
        record{4, 9.0F, 12.0},
        record{15, 13.0F, 14.0},
        record{17, 17.0F, 16.0},
        record{37, 21.0F, 18.0},
    };
    expect_records_eq(records.template to<std::vector>(), expected);
}

TEST(JaggedArrayKernel, StdTupleSoa)
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
    auto records = managed_record_jagged(nested);

    update_soa_rows_kernel<<<2, 4>>>(records);
    synchronize();
    update_soa_flat_data_kernel<<<2, 4>>>(records);
    synchronize();
    update_soa_row_data_kernel<<<2, 4>>>(records);
    synchronize();

    auto expected = std::vector<record>{
        record{4, 9.0F, 12.0},
        record{15, 13.0F, 14.0},
        record{17, 17.0F, 16.0},
        record{37, 21.0F, 18.0},
    };
    expect_records_eq(records.template to<std::vector>(), expected);
}

TEST(JaggedArrayKernel, GpuTupleSoaFlatViews)
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

    gpu_array_test::launch_input_range<update_soa_flat_views_kernel<decltype(records)>>({2}, {3}, records);
    synchronize();

    expect_records_eq(records.template to<std::vector>(),
                      std::vector<record>{record{1, 1.5F, 2.5}, record{102, 2.5F, 3.5}, record{203, 3.5F, 4.5},
                                          record{304, 4.5F, 5.5}});
}

TEST(JaggedArrayKernel, GpuTupleSoaRowViews)
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

    update_soa_row_views_kernel<<<2, 3>>>(records);
    synchronize();

    expect_records_eq(
        records.template to<std::vector>(),
        std::vector<record>{record{1, 1.5F, 2.5}, record{2, 12.5F, 3.5}, record{3, 14.5F, 4.5}, record{4, 34.5F, 5.5}});
}

// NOLINTEND
