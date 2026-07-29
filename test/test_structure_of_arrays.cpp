#include "gpu_array.hpp"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
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

    struct recursive_memory_ops_element
    {
        mutable int prefetch_count = 0;
        mutable int prefetch_device_id = gpuInvalidDeviceId;
        mutable int mem_advise_count = 0;
        mutable int mem_advise_device_id = gpuInvalidDeviceId;
        mutable gpu_array::api::gpuMemoryAdvise mem_advise_value = gpu_array::api::gpuMemoryAdvise::UnsetReadMostly;

        void prefetch(int device_id, gpu_array::api::gpuStream_t) const
        {
            ++prefetch_count;
            prefetch_device_id = device_id;
        }

        void mem_advise(gpu_array::api::gpuMemoryAdvise advise, int device_id) const
        {
            ++mem_advise_count;
            mem_advise_device_id = device_id;
            mem_advise_value = advise;
        }
    };

    template <typename T>
    concept has_use_count = requires(T value) { value.use_count(); };

    template <typename T>
    concept has_reset = requires(T value) { value.reset(); };

    template <typename T>
    concept can_make_view = requires(T&& value) { std::forward<T>(value).view(); };

    template <typename Record>
    void expect_record_eq(const Record& record, int a, float b, double c)
    {
        EXPECT_EQ(record.get_a(), a);
        EXPECT_FLOAT_EQ(record.get_b(), b);
        EXPECT_DOUBLE_EQ(record.get_c(), c);
    }

    template <typename Record>
    std::vector<Record> make_records()
    {
        return {Record{1, 1.5F, 2.5}, Record{2, 3.5F, 4.5}, Record{3, 5.5F, 6.5}, Record{4, 7.5F, 8.5}};
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

    template <typename Soa>
    void check_soa_host_conversions()
    {
        using record = typename Soa::value_type;
        const auto expected = make_records<record>();
        const auto value = Soa(expected);

        expect_records_eq(value.template to<std::vector>(), expected);

        const auto fixed_size = value.template to<std::array<record, 4>>();
        for (auto index = std::size_t{0}; index < fixed_size.size(); ++index)
        {
            expect_record_eq(fixed_size[index], expected[index].get_a(), expected[index].get_b(),
                             expected[index].get_c());
        }
        EXPECT_THROW((value.template to<std::array<record, 3>>()), std::runtime_error);

        const auto explicit_vector = static_cast<std::vector<record>>(value);
        expect_records_eq(explicit_vector, expected);

        const auto empty = Soa();
        EXPECT_TRUE(empty.template to<std::vector>().empty());
    }

    template <typename Soa>
    constexpr void check_soa_range_concepts()
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        static_assert(std::ranges::range<Soa>);
#if defined(GPU_DEVICE_COMPILE)
        static_assert(std::ranges::borrowed_range<Soa>);
#else
        static_assert(!std::ranges::borrowed_range<Soa>);
#endif
        static_assert(std::ranges::view<Soa>);
        static_assert(std::ranges::input_range<Soa>);
        static_assert(std::ranges::forward_range<Soa>);
        static_assert(std::ranges::bidirectional_range<Soa>);
        static_assert(std::ranges::random_access_range<Soa>);
        static_assert(std::ranges::sized_range<Soa>);
        static_assert(std::ranges::common_range<Soa>);
        static_assert(std::ranges::viewable_range<Soa>);
        static_assert(!std::ranges::contiguous_range<Soa>);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
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

TEST(StructureOfArrays, RangeTypes)
{
    using record = gpu_tuple_record<int, float, double>;
    using device_soa = gpu_array::structure_of_arrays<record, std::uint32_t>;
    using managed_soa = gpu_array::managed_structure_of_arrays<record, std::uint32_t>;
    using grid_thread_strided_view = decltype(std::declval<managed_soa&>() | gpu_array::views::grid_thread_stride);
    using enumerate_soa_view = gpu_array::enumerate_view<managed_soa>;
    using zip_soa_view = gpu_array::zip_view<managed_soa, managed_soa>;
    using const_zip_device_soa_view =
        decltype(gpu_array::views::zip(std::declval<device_soa&>(), std::declval<const device_soa&>()));
    using const_zip_soa_view =
        decltype(gpu_array::views::zip(std::declval<managed_soa&>(), std::declval<const managed_soa&>()));

    static_assert(std::same_as<typename device_soa::size_type, std::uint32_t>);
    static_assert(std::same_as<typename device_soa::template element_type<1>, float>);
    static_assert(std::same_as<std::ranges::range_value_t<device_soa>, record>);
    static_assert(std::same_as<std::ranges::range_reference_t<device_soa>, gpu_tuple_record<int&, float&, double&>>);
    static_assert(std::same_as<std::ranges::range_reference_t<const device_soa>,
                               gpu_tuple_record<const int&, const float&, const double&>>);
    static_assert(std::same_as<std::ranges::range_value_t<managed_soa>, record>);
    static_assert(std::same_as<std::ranges::range_reference_t<managed_soa>, gpu_tuple_record<int&, float&, double&>>);
    static_assert(std::same_as<decltype(std::declval<device_soa&>().template data<1>()), float*>);
    static_assert(std::same_as<decltype(std::declval<const device_soa&>().template data<1>()), const float*>);
    static_assert(std::ranges::input_range<grid_thread_strided_view>);
    static_assert(std::same_as<std::ranges::range_value_t<grid_thread_strided_view>, record>);
    static_assert(std::same_as<std::ranges::range_reference_t<grid_thread_strided_view>,
                               gpu_tuple_record<int&, float&, double&>>);
    static_assert(std::ranges::input_range<enumerate_soa_view>);
    static_assert(
        std::same_as<std::ranges::range_value_t<enumerate_soa_view>, gpu_array::tuple<std::uint32_t, record>>);
    static_assert(std::same_as<std::ranges::range_reference_t<enumerate_soa_view>,
                               gpu_array::tuple<std::uint32_t, gpu_tuple_record<int&, float&, double&>>>);
    static_assert(std::ranges::input_range<zip_soa_view>);
    static_assert(std::same_as<std::ranges::range_value_t<zip_soa_view>, gpu_array::tuple<record, record>>);
    static_assert(std::same_as<
                  std::ranges::range_reference_t<zip_soa_view>,
                  gpu_array::tuple<gpu_tuple_record<int&, float&, double&>, gpu_tuple_record<int&, float&, double&>>>);
    static_assert(std::same_as<std::ranges::range_reference_t<const_zip_device_soa_view>,
                               gpu_array::tuple<gpu_tuple_record<int&, float&, double&>,
                                                gpu_tuple_record<const int&, const float&, const double&>>>);
    static_assert(std::same_as<std::ranges::range_reference_t<const_zip_soa_view>,
                               gpu_array::tuple<gpu_tuple_record<int&, float&, double&>,
                                                gpu_tuple_record<const int&, const float&, const double&>>>);
    static_assert(
        requires(managed_soa& values) { values | gpu_array::views::enumerate | gpu_array::views::grid_thread_stride; });
    static_assert(requires(managed_soa& lhs, managed_soa& rhs) {
        gpu_array::views::zip(lhs, rhs) | gpu_array::views::grid_thread_stride;
    });

    check_soa_range_concepts<device_soa>();
    check_soa_range_concepts<managed_soa>();

    SUCCEED();
}

TEST(StructureOfArrays, ConstViewTypesAndMaterialization)
{
    using record = gpu_tuple_record<int, float, double>;
    using const_view = gpu_array::const_managed_structure_of_arrays_view<record, std::uint32_t>;

    static_assert(std::same_as<const_view::value_type, record>);
    static_assert(std::same_as<const_view::reference, gpu_tuple_record<const int&, const float&, const double&>>);
    static_assert(std::same_as<const_view::iterator,
                               gpu_array::structure_of_arrays_iterator<gpu_tuple_record, const int, const float,
                                                                      const double>>);
    static_assert(std::same_as<std::ranges::range_value_t<const_view>, record>);
    static_assert(std::same_as<std::ranges::range_reference_t<const_view>,
                               gpu_tuple_record<const int&, const float&, const double&>>);
    static_assert(!std::ranges::output_range<const_view, record>);
    static_assert(std::same_as<decltype(std::declval<const const_view&>().template to<std::vector>()),
                               std::vector<record>>);

    const auto expected = make_records<record>();
    const auto source = gpu_array::managed_structure_of_arrays<record>(expected);
    const auto view = source.view();
    expect_records_eq(view.template to<std::vector>(), expected);
}

TEST(StructureOfArraysView, PublicContractAndOwnerFactories)
{
    using record = gpu_tuple_record<int, float, double>;
    using device_soa = gpu_array::structure_of_arrays<record, std::uint32_t>;
    using managed_soa = gpu_array::managed_structure_of_arrays<record, std::uint32_t>;
    using device_view = gpu_array::structure_of_arrays_view<record, std::uint32_t>;
    using managed_view = gpu_array::managed_structure_of_arrays_view<record, std::uint32_t>;
    using const_device_view = gpu_array::const_structure_of_arrays_view<record, std::uint32_t>;
    using const_managed_view = gpu_array::const_managed_structure_of_arrays_view<record, std::uint32_t>;

    static_assert(std::ranges::view<device_view>);
    static_assert(std::ranges::random_access_range<device_view>);
    static_assert(std::ranges::view<managed_view>);
    static_assert(std::ranges::random_access_range<managed_view>);
    static_assert(std::same_as<std::ranges::range_reference_t<const_device_view>,
                               gpu_tuple_record<const int&, const float&, const double&>>);
    static_assert(std::same_as<std::ranges::range_reference_t<const_managed_view>,
                               gpu_tuple_record<const int&, const float&, const double&>>);
    static_assert(std::same_as<decltype(std::declval<const const_device_view&>().template data<0>()), const int*>);
    static_assert(
        std::same_as<decltype(std::declval<const const_managed_view&>().template data<1>()), const float*>);
    static_assert(!gpu_array::detail::gpu_array_ptr<device_view>);
    static_assert(!gpu_array::detail::gpu_array_ptr<managed_view>);
    static_assert(!has_use_count<device_view>);
    static_assert(!has_reset<managed_view>);
    static_assert(!can_make_view<device_soa&&>);
    static_assert(!can_make_view<const device_soa&&>);
    static_assert(!can_make_view<managed_soa&&>);
    static_assert(!can_make_view<const managed_soa&&>);

    const auto expected = make_records<record>();
    auto device_values = device_soa(expected);
    auto managed_values = managed_soa(expected);
    const auto& const_device_values = device_values;
    const auto& const_managed_values = managed_values;

    static_assert(std::same_as<decltype(device_values.view()), device_view>);
    static_assert(std::same_as<decltype(const_device_values.view()), const_device_view>);
    static_assert(std::same_as<decltype(managed_values.view()), managed_view>);
    static_assert(std::same_as<decltype(const_managed_values.view()), const_managed_view>);

    const auto device_slice = device_values.view(1, 2);
    EXPECT_EQ(device_slice.template data<0>(), device_values.template data<0>() + 1);
    EXPECT_EQ(device_slice.template data<1>(), device_values.template data<1>() + 1);
    EXPECT_EQ(device_slice.template data<2>(), device_values.template data<2>() + 1);
    expect_records_eq(device_slice.template to<std::vector>(), {expected[1], expected[2]});
    EXPECT_TRUE(device_values.view(0, 0).empty());
    EXPECT_EQ(const_device_values.view(device_values.size(), 0).template data<0>(), nullptr);

    auto managed_slice = managed_values.view(1, 2);
    managed_slice[0].get_a() = 20;
    managed_slice[1].get_c() = 70.0;
    expect_records_eq(managed_values.template to<std::vector>(),
                      {expected[0], record{20, 3.5F, 4.5}, record{3, 5.5F, 70.0}, expected[3]});
    expect_records_eq(const_managed_values.view(2, 1).template to<std::vector>(), {record{3, 5.5F, 70.0}});

    auto empty_owner = managed_soa{};
    EXPECT_TRUE(empty_owner.view().empty());
    EXPECT_EQ(empty_owner.view().template data<0>(), nullptr);
}

TEST(StructureOfArraysView, ValidatesMemoryKind)
{
    using record = gpu_tuple_record<int, float, double>;
    auto device_values = gpu_array::structure_of_arrays<record, std::uint32_t>(make_records<record>());
    auto managed_values = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(make_records<record>());

    EXPECT_THROW((gpu_array::structure_of_arrays_view<record, std::uint32_t>(
                     managed_values.template data<0>(), managed_values.template data<1>(),
                     managed_values.template data<2>(), managed_values.size())),
                 std::runtime_error);
    EXPECT_THROW((gpu_array::structure_of_arrays_view<record, std::uint32_t>(
                     device_values.template data<0>(), managed_values.template data<1>(),
                     device_values.template data<2>(), device_values.size())),
                 std::runtime_error);
    EXPECT_THROW((gpu_array::managed_structure_of_arrays_view<record, std::uint32_t>(
                     device_values.template data<0>(), device_values.template data<1>(),
                     device_values.template data<2>(), device_values.size())),
                 std::runtime_error);
}

TEST(StructureOfArrays, TupleInput)
{
    using gpu_pair = gpu_array::tuple<int, float>;
    using std_pair = std::tuple<int, float>;
    using gpu_record = gpu_tuple_record<int, float, double>;
    using std_record = std_tuple_record<int, float, double>;
    using triple = gpu_array::tuple<int, float, double>;

    static_assert(std::constructible_from<gpu_array::structure_of_arrays<gpu_pair>, const std::vector<gpu_pair>&>);
    static_assert(
        std::constructible_from<gpu_array::managed_structure_of_arrays<std_pair>, const std::vector<std_pair>&>);
    static_assert(
        std::constructible_from<gpu_array::managed_structure_of_arrays<gpu_record>, const std::vector<gpu_record>&>);
    static_assert(
        std::constructible_from<gpu_array::managed_structure_of_arrays<std_record>, const std::vector<std_record>&>);

    static_assert(
        !std::constructible_from<gpu_array::managed_structure_of_arrays<triple>, const std::vector<gpu_pair>&>);
    static_assert(
        !std::constructible_from<gpu_array::managed_structure_of_arrays<triple>, const std::vector<std_pair>&>);
    static_assert(
        !std::constructible_from<gpu_array::managed_structure_of_arrays<gpu_pair>, const std::vector<triple>&>);

    using shorthand = gpu_array::managed_structure_of_arrays<int, float>;
    static_assert(std::same_as<std::ranges::range_value_t<shorthand>, gpu_array::tuple<int, float>>);

    SUCCEED();
}

TEST(StructureOfArrays, HostConversions)
{
    using record = gpu_tuple_record<int, float, double>;
    check_soa_host_conversions<gpu_array::structure_of_arrays<record, std::uint32_t>>();
    check_soa_host_conversions<gpu_array::managed_structure_of_arrays<record, std::uint32_t>>();
}

TEST(ManagedStructureOfArrays, GpuTupleRecord)
{
    using record = gpu_tuple_record<int, float, double>;
    const auto source = std::vector<record>{record{1, 2.0F, 3.0}, record{4, 5.0F, 6.0}};

    auto value = gpu_array::managed_structure_of_arrays<record>(source);
    ASSERT_EQ(value.size(), 2U);
    expect_record_eq(value[0], 1, 2.0F, 3.0);
    expect_record_eq(value[1], 4, 5.0F, 6.0);

    value[1].get_a() = 7;
    value[1].get_b() = 8.0F;
    value[1].get_c() = 9.0;

    const auto round_trip = value.template to<std::vector>();
    ASSERT_EQ(round_trip.size(), 2U);
    expect_record_eq(round_trip[0], 1, 2.0F, 3.0);
    expect_record_eq(round_trip[1], 7, 8.0F, 9.0);
}

TEST(ManagedStructureOfArrays, StdTupleRecord)
{
    using record = std_tuple_record<int, float, double>;
    const auto source = std::vector<record>{record{2, 3.0F, 4.0}, record{5, 6.0F, 7.0}};

    auto value = gpu_array::managed_structure_of_arrays<record>(source);
    ASSERT_EQ(value.size(), 2U);
    expect_record_eq(value[0], 2, 3.0F, 4.0);
    expect_record_eq(value[1], 5, 6.0F, 7.0);

    value[0].get_a() = 8;
    value[0].get_b() = 9.0F;
    value[0].get_c() = 10.0;

    const auto round_trip = value.template to<std::vector>();
    ASSERT_EQ(round_trip.size(), 2U);
    expect_record_eq(round_trip[0], 8, 9.0F, 10.0);
    expect_record_eq(round_trip[1], 5, 6.0F, 7.0);
}

TEST(ManagedStructureOfArrays, GridThreadStrideView)
{
    using record = gpu_tuple_record<int, float, double>;
    auto records = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(make_records<record>());

    for (auto&& value : records | gpu_array::views::grid_thread_stride)
    {
        value.get_a() = value.get_a() * 2 + 1;
        value.get_b() += 0.5F;
        value.get_c() += 1.0;
    }

    expect_records_eq(
        records.template to<std::vector>(),
        std::vector<record>{record{3, 2.0F, 3.5}, record{5, 4.0F, 5.5}, record{7, 6.0F, 7.5}, record{9, 8.0F, 9.5}});
}

TEST(ManagedStructureOfArrays, EnumerateViewReferences)
{
    using record = gpu_tuple_record<int, float, double>;
    auto records = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(make_records<record>());

    for (auto&& [index, value] : records | gpu_array::views::enumerate)
    {
        value.get_a() += static_cast<int>(index) * 10;
        value.get_b() += static_cast<float>(index);
        value.get_c() += static_cast<double>(index) * 0.25;
    }

    expect_records_eq(records.template to<std::vector>(),
                      std::vector<record>{record{1, 1.5F, 2.5}, record{12, 4.5F, 4.75}, record{23, 7.5F, 7.0},
                                          record{34, 10.5F, 9.25}});
}

TEST(ManagedStructureOfArrays, ZipViewReferences)
{
    using record = gpu_tuple_record<int, float, double>;
    auto lhs = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(make_records<record>());
    auto rhs = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(
        std::vector<record>{record{10, 10.0F, 10.0}, record{20, 20.0F, 20.0}, record{30, 30.0F, 30.0}});

    for (auto&& [left, right] : gpu_array::views::zip(lhs, rhs))
    {
        left.get_a() += right.get_a();
        left.get_b() += right.get_b();
        left.get_c() += right.get_c();
    }

    expect_records_eq(lhs.template to<std::vector>(),
                      std::vector<record>{record{11, 11.5F, 12.5}, record{22, 23.5F, 24.5}, record{33, 35.5F, 36.5},
                                          record{4, 7.5F, 8.5}});
}

TEST(ManagedStructureOfArrays, RecursiveMemoryManagement)
{
    using record =
        gpu_tuple_record<recursive_memory_ops_element, recursive_memory_ops_element, recursive_memory_ops_element>;
    auto values = gpu_array::managed_structure_of_arrays<record, std::uint32_t>(3, gpu_array::default_init);
    auto view = values.view();
    auto device_id = 0;
    GPU_CHECK_ERROR(gpu_array::api::gpuGetDevice(&device_id));

    view.prefetch(device_id, 0, false);
    EXPECT_EQ(values[0].get_a().prefetch_count, 0);
    EXPECT_EQ(values[0].get_b().prefetch_count, 0);
    EXPECT_EQ(values[0].get_c().prefetch_count, 0);

    view.prefetch(1, 2, device_id, 0, true);
    EXPECT_EQ(values[0].get_a().prefetch_count, 0);
    EXPECT_EQ(values[1].get_a().prefetch_count, 1);
    EXPECT_EQ(values[1].get_b().prefetch_count, 1);
    EXPECT_EQ(values[1].get_c().prefetch_count, 1);
    EXPECT_EQ(values[2].get_a().prefetch_device_id, device_id);
    EXPECT_EQ(values[2].get_b().prefetch_device_id, device_id);
    EXPECT_EQ(values[2].get_c().prefetch_device_id, device_id);

    view.prefetch_to_cpu(1, 1, 0, true);
    EXPECT_EQ(values[0].get_a().prefetch_count, 0);
    EXPECT_EQ(values[1].get_a().prefetch_count, 2);
    EXPECT_EQ(values[1].get_b().prefetch_count, 2);
    EXPECT_EQ(values[1].get_c().prefetch_count, 2);
    EXPECT_EQ(values[1].get_a().prefetch_device_id, gpuCpuDeviceId);
    EXPECT_EQ(values[1].get_b().prefetch_device_id, gpuCpuDeviceId);
    EXPECT_EQ(values[1].get_c().prefetch_device_id, gpuCpuDeviceId);

    view.mem_advise(gpu_array::api::gpuMemoryAdvise::SetReadMostly, device_id, false);
    EXPECT_EQ(values[0].get_a().mem_advise_count, 0);
    EXPECT_EQ(values[1].get_b().mem_advise_count, 0);
    EXPECT_EQ(values[2].get_c().mem_advise_count, 0);

    view.mem_advise(0, 2, gpu_array::api::gpuMemoryAdvise::SetReadMostly, device_id, true);
    EXPECT_EQ(values[0].get_a().mem_advise_count, 1);
    EXPECT_EQ(values[0].get_b().mem_advise_count, 1);
    EXPECT_EQ(values[0].get_c().mem_advise_count, 1);
    EXPECT_EQ(values[1].get_a().mem_advise_device_id, device_id);
    EXPECT_EQ(values[1].get_b().mem_advise_device_id, device_id);
    EXPECT_EQ(values[1].get_c().mem_advise_device_id, device_id);
    EXPECT_EQ(values[1].get_a().mem_advise_value, gpu_array::api::gpuMemoryAdvise::SetReadMostly);
    EXPECT_EQ(values[2].get_a().mem_advise_count, 0);

    view.mem_advise_to_cpu(1, 1, gpu_array::api::gpuMemoryAdvise::SetPreferredLocation, true);
    EXPECT_EQ(values[0].get_a().mem_advise_count, 1);
    EXPECT_EQ(values[1].get_a().mem_advise_count, 2);
    EXPECT_EQ(values[1].get_b().mem_advise_count, 2);
    EXPECT_EQ(values[1].get_c().mem_advise_count, 2);
    EXPECT_EQ(values[1].get_a().mem_advise_device_id, gpuCpuDeviceId);
    EXPECT_EQ(values[1].get_b().mem_advise_device_id, gpuCpuDeviceId);
    EXPECT_EQ(values[1].get_c().mem_advise_device_id, gpuCpuDeviceId);
    EXPECT_EQ(values[1].get_a().mem_advise_value, gpu_array::api::gpuMemoryAdvise::SetPreferredLocation);
}

// NOLINTEND
