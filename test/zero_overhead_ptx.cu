#include "gpu_array.hpp"

#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>

namespace
{
    using gpu_array::managed_value;
    using gpu_array::managed_array;
    using gpu_array::value;
    namespace views = gpu_array::views;
}

// Baseline block-thread stride pattern used by the README zero-overhead example.
__global__ void ga_ptx_raw_pointer_kernel(int* data, std::uint32_t size)
{
    const auto block = cooperative_groups::this_thread_block();
    for (auto index = static_cast<std::uint32_t>(block.thread_rank()); index < size;
         index += static_cast<std::uint32_t>(block.size()))
    {
        data[index] = 1;
    }
}

__global__ void ga_ptx_array_kernel(managed_array<int, std::uint32_t> array)
{
    const auto block = cooperative_groups::this_thread_block();
    for (auto index = static_cast<std::uint32_t>(block.thread_rank()); index < array.size();
         index += static_cast<std::uint32_t>(block.size()))
    {
        array[index] = 1;
    }
}

__global__ void ga_ptx_array_view_kernel(managed_array<int, std::uint32_t> array)
{
    for (auto& value : array | views::block_thread_stride)
    {
        value = 1;
    }
}

// Grid-thread stride is the common full-grid variant of the same raw-pointer loop pattern.
__global__ void ga_ptx_raw_grid_stride_kernel(int* data, std::uint32_t size)
{
    const auto grid = cooperative_groups::this_grid();
    for (auto index = static_cast<std::uint32_t>(grid.thread_rank()); index < size;
         index += static_cast<std::uint32_t>(grid.size()))
    {
        data[index] = 1;
    }
}

__global__ void ga_ptx_grid_stride_view_kernel(managed_array<int, std::uint32_t> array)
{
    for (auto& element : array | views::grid_thread_stride)
    {
        element = 1;
    }
}

// Enumerate should compile to the same index arithmetic as a hand-written grid-stride loop.
__global__ void ga_ptx_raw_enumerate_kernel(int* data, std::uint32_t size)
{
    const auto grid = cooperative_groups::this_grid();
    for (auto index = static_cast<std::uint32_t>(grid.thread_rank()); index < size;
         index += static_cast<std::uint32_t>(grid.size()))
    {
        data[index] += static_cast<int>(index);
    }
}

__global__ void ga_ptx_enumerate_view_kernel(managed_array<int, std::uint32_t> array)
{
    for (auto&& [index, value] : array | views::enumerate | views::grid_thread_stride)
    {
        value += static_cast<int>(index);
    }
}

// Zip should add no loop-body work beyond the manual shorter-range traversal.
__global__ void ga_ptx_raw_zip_kernel(int* lhs, const int* rhs, std::uint32_t lhs_size, std::uint32_t rhs_size)
{
    const auto size = std::min(lhs_size, rhs_size);
    const auto grid = cooperative_groups::this_grid();
    for (auto index = static_cast<std::uint32_t>(grid.thread_rank()); index < size;
         index += static_cast<std::uint32_t>(grid.size()))
    {
        lhs[index] += rhs[index];
    }
}

__global__ void ga_ptx_zip_view_kernel(managed_array<int, std::uint32_t> lhs, managed_array<int, std::uint32_t> rhs)
{
    for (auto&& [left, right] : views::zip(lhs, rhs) | views::grid_thread_stride)
    {
        left += right;
    }
}

// Check zip followed by enumerate against the same raw-pointer formula.
__global__ void ga_ptx_raw_zip_enumerate_kernel(int* lhs, const int* rhs, std::uint32_t lhs_size,
                                                std::uint32_t rhs_size)
{
    const auto size = std::min(lhs_size, rhs_size);
    const auto grid = cooperative_groups::this_grid();
    for (auto index = static_cast<std::uint32_t>(grid.thread_rank()); index < size;
         index += static_cast<std::uint32_t>(grid.size()))
    {
        lhs[index] = lhs[index] * 100 + rhs[index] * static_cast<int>(index + 1);
    }
}

__global__ void ga_ptx_zip_enumerate_view_kernel(managed_array<int, std::uint32_t> lhs,
                                                 managed_array<int, std::uint32_t> rhs)
{
    for (auto&& [index, zipped] : views::zip(lhs, rhs) | views::enumerate | views::grid_thread_stride)
    {
        auto&& [left, right] = zipped;
        left = left * 100 + right * static_cast<int>(index + 1);
    }
}

// Single-value wrappers should compile to the same dereference as a raw pointer.
__global__ void ga_ptx_raw_value_kernel(int* data, int offset)
{
    *data += offset;
}

__global__ void ga_ptx_value_kernel(value<int> data, int offset)
{
    *data += offset;
}

__global__ void ga_ptx_managed_value_kernel(managed_value<int> data, int offset)
{
    *data += offset;
}
