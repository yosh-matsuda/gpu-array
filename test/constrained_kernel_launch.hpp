#pragma once

#include <concepts>
#include <ranges>

namespace gpu_array_test
{
    // ROCm clang can reject constrained __global__ template launches even when the same std::ranges
    // concepts are satisfied. These helpers keep the concept checks on ordinary host functions and
    // launch unconstrained kernel specializations from there.
    template <typename Range>
    concept input_range = std::ranges::input_range<Range>;

    template <typename Range>
    concept nested_input_range = input_range<Range> && std::ranges::input_range<std::ranges::range_value_t<Range>>;

    template <typename Lhs, typename Rhs>
    concept input_range_pair = input_range<Lhs> && input_range<Rhs>;

    template <typename Lhs, typename Rhs>
    concept nested_input_range_pair = input_range_pair<Lhs, Rhs> &&
                                      std::ranges::input_range<std::ranges::range_value_t<Lhs>> &&
                                      std::ranges::input_range<std::ranges::range_value_t<Rhs>>;

    template <auto Kernel, input_range Range, typename... Args>
    void launch_input_range(dim3 grid, dim3 block, Range range, Args... args)
    {
        Kernel<<<grid, block>>>(range, args...);
    }

    template <auto Kernel, nested_input_range Range, typename... Args>
    void launch_nested_input_range(dim3 grid, dim3 block, Range range, Args... args)
    {
        Kernel<<<grid, block>>>(range, args...);
    }

    template <auto Kernel, typename Lhs, typename Rhs, typename... Args>
    requires input_range_pair<Lhs, Rhs>
    void launch_input_range_pair(dim3 grid, dim3 block, Lhs lhs, Rhs rhs, Args... args)
    {
        Kernel<<<grid, block>>>(lhs, rhs, args...);
    }

    template <auto Kernel, typename Lhs, typename Rhs, typename... Args>
    requires nested_input_range_pair<Lhs, Rhs>
    void launch_nested_input_range_pair(dim3 grid, dim3 block, Lhs lhs, Rhs rhs, Args... args)
    {
        Kernel<<<grid, block>>>(lhs, rhs, args...);
    }
}  // namespace gpu_array_test
