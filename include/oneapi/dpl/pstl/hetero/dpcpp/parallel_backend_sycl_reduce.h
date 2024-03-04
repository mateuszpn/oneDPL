// -*- C++ -*-
//===-- parallel_backend_sycl_reduce.h --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename... _Name>
class __reduce_small_kernel;

template <typename... _Name>
class __reduce_mid_device_kernel;

template <typename... _Name>
class __reduce_mid_work_group_kernel;

template <typename... _Name>
class __reduce_kernel;

// Adjust number of sequential operations per work items based on the vector size. Single elements are kept to
// improve performance of small arrays or remainder loops.
template <int _VecSize, typename _Size>
void
__adjust_iters_per_work_item(_Size& __iters_per_work_item)
{
    if (__iters_per_work_item > 1)
        __iters_per_work_item = ((__iters_per_work_item + _VecSize - 1) / _VecSize) * _VecSize;
}

// Single work group kernel that transforms and reduces __n elements to the single result.
template <typename _Tp, typename _NDItemId, typename _Size, typename _TransformPattern, typename _ReducePattern,
          typename _InitType, typename _AccLocal, typename _Res, typename... _Acc>
void
__work_group_reduce_kernel(const _NDItemId __item_id, const _Size __n, ::std::uint8_t __iters_per_work_item,
                           _TransformPattern __transform_pattern, _ReducePattern __reduce_pattern, _InitType __init,
                           const _AccLocal& __local_mem, const _Res& __res_acc, const _Acc&... __acc)
{
    auto __local_idx = __item_id.get_local_id(0);
    auto __group_size = __item_id.get_local_range().size();
    // 1. Initialization (transform part). Fill local memory
    __transform_pattern(__item_id, __n, __iters_per_work_item, /*global_offset*/ (_Size)0, __local_mem, __acc...);
    __dpl_sycl::__group_barrier(__item_id);
    const _Size __n_items = __transform_pattern.output_size(__n, __group_size, __iters_per_work_item);
    // 2. Reduce within work group using local memory
    _Tp __result = __reduce_pattern(__item_id, __n_items, __local_mem);
    if (__local_idx == 0)
    {
        __reduce_pattern.apply_init(__init, __result);
        __res_acc[0] = __result;
    }
}

// Device kernel that transforms and reduces __n elements to the number of work groups preliminary results.
template <typename _Tp, typename _NDItemId, typename _Size, typename _TransformPattern, typename _ReducePattern,
          typename _AccLocal, typename _Tmp, typename... _Acc>
_Tp
__device_reduce_kernel(const _NDItemId __item_id, const _Size __n, ::std::uint8_t __iters_per_work_item, int __iter,
                       _TransformPattern __transform_pattern, _ReducePattern __reduce_pattern,
                       const _AccLocal& __local_mem, const _Tmp& __temp_acc, const _Acc&... __acc)
{
    auto __group_size = __item_id.get_local_range().size();
    const _Size __global_offset = __iter * __group_size * __iters_per_work_item;
    // 1. Initialization (transform part). Fill local memory
    __transform_pattern(__item_id, __n, __iters_per_work_item, __global_offset, __local_mem, __acc...);
    __dpl_sycl::__group_barrier(__item_id);
    const _Size __n_items = __transform_pattern.output_size(__n, __group_size, __iters_per_work_item);
    // 2. Reduce within work group using local memory
    return __reduce_pattern(__item_id, __n_items, __local_mem);
}

//------------------------------------------------------------------------
// parallel_transform_reduce - async patterns
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------

// Parallel_transform_reduce for a small arrays using a single work group.
// Transforms and reduces __work_group_size * __iters_per_work_item elements.
template <typename _Tp, typename _Commutative, int _VecSize, typename _KernelName>
struct __parallel_transform_reduce_small_submitter;

template <typename _Tp, typename _Commutative, int _VecSize, typename... _Name>
struct __parallel_transform_reduce_small_submitter<_Tp, _Commutative, _VecSize,
                                                   __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, const _Size __n, ::std::uint16_t __work_group_size,
               ::std::uint8_t __iters_per_work_item, _ReduceOp __reduce_op, _TransformOp __transform_op,
               _InitType __init, _Ranges&&... __rngs) const
    {
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _TransformOp, _Commutative, _VecSize>{
                __reduce_op, __transform_op};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        __usm_host_or_buffer_storage<_ExecutionPolicy, _Tp> __res_container(__exec, 1);

        sycl::event __reduce_event = __exec.queue().submit([&, __n](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            auto __res_acc = __res_container.__get_acc(__cgh);
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    auto __res_ptr = __res_acc.__get_pointer();
                    __work_group_reduce_kernel<_Tp>(__item_id, __n, __iters_per_work_item, __transform_pattern,
                                                    __reduce_pattern, __init, __temp_local, __res_ptr, __rngs...);
                });
        });

        return __future(__reduce_event, __res_container);
    }
}; // struct __parallel_transform_reduce_small_submitter

template <typename _Tp, typename _Commutative, int _VecSize, typename _ExecutionPolicy, typename _Size,
          typename _ReduceOp, typename _TransformOp, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_small_impl(_ExecutionPolicy&& __exec, const _Size __n, ::std::uint16_t __work_group_size,
                                       ::std::uint8_t __iters_per_work_item, _ReduceOp __reduce_op,
                                       _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _ReduceKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__reduce_small_kernel<_CustomName>>;

    return __parallel_transform_reduce_small_submitter<_Tp, _Commutative, _VecSize, _ReduceKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), __n, __work_group_size, __iters_per_work_item, __reduce_op,
        __transform_op, __init, ::std::forward<_Ranges>(__rngs)...);
}

// Submits the first kernel of the parallel_transform_reduce for mid-sized arrays.
// Uses multiple work groups that each reduce __work_group_size * __iters_per_work_item items and store the preliminary
// results in __temp.
template <typename _Tp, typename _Commutative, int _VecSize, typename _KernelName>
struct __parallel_transform_reduce_device_kernel_submitter;

template <typename _Tp, typename _Commutative, int _VecSize, typename... _KernelName>
struct __parallel_transform_reduce_device_kernel_submitter<_Tp, _Commutative, _VecSize,
                                                           __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, _Size __n, ::std::uint16_t __work_group_size,
               ::std::uint8_t __iters_per_work_item, _ReduceOp __reduce_op, _TransformOp __transform_op,
               sycl::buffer<_Tp>& __temp, _Ranges&&... __rngs) const
    {
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _TransformOp, _Commutative, _VecSize>{
                __reduce_op, __transform_op};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        // number of buffer elements processed within workgroup
        _Size __size_per_work_group = __iters_per_work_item * __work_group_size;
        ::std::size_t __max_compute_units = oneapi::dpl::__internal::__max_compute_units(__exec);
        const _Size __num_iters =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __max_compute_units * __size_per_work_group);

        return __exec.queue().submit([&, __n, __num_iters](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            sycl::accessor __temp_acc{__temp, __cgh, sycl::write_only, __dpl_sycl::__no_init{}};
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<_KernelName...>(
                sycl::nd_range<1>(sycl::range<1>(__max_compute_units * __work_group_size),
                                  sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    auto __local_idx = __item_id.get_local_id(0);
                    auto __group_idx = __item_id.get_group(0);
                    _Tp __result = __device_reduce_kernel<_Tp>(__item_id, __n, __iters_per_work_item, /*__iter*/ 0,
                                                               __transform_pattern, __reduce_pattern, __temp_local,
                                                               __temp_acc, __rngs...);
                    if (__local_idx == 0)
                        __temp_acc[__group_idx] = __result;
                    for (int __iter = 1; __iter < __num_iters; ++__iter)
                    {
                        __result = __device_reduce_kernel<_Tp>(__item_id, __n, __iters_per_work_item, __iter,
                                                               __transform_pattern, __reduce_pattern, __temp_local,
                                                               __temp_acc, __rngs...);
                        if (__local_idx == 0)
                            __temp_acc[__group_idx] = __reduce_op(__temp_acc[__group_idx], __result);
                    }
                });
        });
    }
}; // struct __parallel_transform_reduce_device_kernel_submitter

// Submits the second kernel of the parallel_transform_reduce for mid-sized arrays.
// Uses a single work groups to reduce __n preliminary results stored in __temp and returns a future object with the
// result buffer.
template <typename _Tp, typename _Commutative, int _VecSize, typename _KernelName>
struct __parallel_transform_reduce_work_group_kernel_submitter;

template <typename _Tp, typename _Commutative, int _VecSize, typename... _KernelName>
struct __parallel_transform_reduce_work_group_kernel_submitter<_Tp, _Commutative, _VecSize,
                                                               __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
    auto
    operator()(_ExecutionPolicy&& __exec, sycl::event& __reduce_event, _Size __n, ::std::uint16_t __work_group_size,
               ::std::uint8_t __iters_per_work_item, _ReduceOp __reduce_op, _InitType __init,
               sycl::buffer<_Tp>& __temp) const
    {
        using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _NoOpFunctor, _Commutative, _VecSize>{
                __reduce_op, _NoOpFunctor{}};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        // Lower the work group size of the second kernel to the next power of 2 if __n < __work_group_size.
        auto __work_group_size2 = __work_group_size;
        if (__iters_per_work_item == 1)
        {
            if (__n < __work_group_size)
            {
                __work_group_size2 = __n;
                if ((__work_group_size2 & (__work_group_size2 - 1)) != 0)
                    __work_group_size2 = oneapi::dpl::__internal::__dpl_bit_floor(__work_group_size2) << 1;
            }
        }

        __usm_host_or_buffer_storage<_ExecutionPolicy, _Tp> __res_container(__exec, 1);

        __reduce_event = __exec.queue().submit([&, __n](sycl::handler& __cgh) {
            __cgh.depends_on(__reduce_event);

            sycl::accessor __temp_acc{__temp, __cgh, sycl::read_only};
            auto __res_acc = __res_container.__get_acc(__cgh);
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);

            __cgh.parallel_for<_KernelName...>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size2), sycl::range<1>(__work_group_size2)),
                [=](sycl::nd_item<1> __item_id) {
                    auto __res_ptr = __res_acc.__get_pointer();
                    __work_group_reduce_kernel<_Tp>(__item_id, __n, __iters_per_work_item, __transform_pattern,
                                                    __reduce_pattern, __init, __temp_local, __res_ptr, __temp_acc);
                });
        });

        return __future(__reduce_event, __res_container);
    }
}; // struct __parallel_transform_reduce_work_group_kernel_submitter

template <typename _Tp, typename _Commutative, int _VecSize, typename _ExecutionPolicy, typename _Size,
          typename _ReduceOp, typename _TransformOp, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_impl(_ExecutionPolicy&& __exec, _Size __n, ::std::uint16_t __work_group_size,
                                 ::std::uint8_t __iters_per_work_item_device_kernel,
                                 ::std::uint8_t __iters_per_work_item_work_group_kernel, _ReduceOp __reduce_op,
                                 _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _ReduceDeviceKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__reduce_mid_device_kernel<_CustomName>>;
    using _ReduceWorkGroupKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_mid_work_group_kernel<_CustomName>>;

    // number of buffer elements processed within workgroup
    _Size __size_per_work_group = __iters_per_work_item_device_kernel * __work_group_size;
    const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
    sycl::buffer<_Tp> __temp{sycl::range<1>(__n_groups)};

    sycl::event __reduce_event =
        __parallel_transform_reduce_device_kernel_submitter<_Tp, _Commutative, _VecSize, _ReduceDeviceKernel>()(
            __exec, __n, __work_group_size, __iters_per_work_item_device_kernel, __reduce_op, __transform_op, __temp,
            ::std::forward<_Ranges>(__rngs)...);

    __n = __n_groups; // Number of preliminary results from the device kernel.
    return __parallel_transform_reduce_work_group_kernel_submitter<_Tp, _Commutative, _VecSize,
                                                                   _ReduceWorkGroupKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), __reduce_event, __n, __work_group_size,
        __iters_per_work_item_work_group_kernel, __reduce_op, __init, __temp);
}

// General version of parallel_transform_reduce.
// The binary operator must be associative but commutativity is only required by some of the algorithms using
// __parallel_transform_reduce. This is provided by the _Commutative parameter. The current implementation uses a
// generic implementation that processes elements in order. However, future improvements might be possible utilizing
// the commutative property of the respective algorithms.
//
// Each work item transforms and reduces __iters_per_work_item elements from global memory and stores the result in SLM.
// 32 __iters_per_work_item was empirically found best for typical devices.
// Each work group of size __work_group_size reduces the preliminary results of each work item in a group reduction
// using SLM. 256 __work_group_size was empirically found best for typical devices.
// A single-work group implementation is used for small arrays.
// Mid-sized arrays use two tree reductions with independent __iters_per_work_item.
// Big arrays are processed with a recursive tree reduction. __work_group_size * __iters_per_work_item elements are
// reduced in each step.
template <typename _Tp, typename _Commutative, int _VecSize, typename _ExecutionPolicy, typename _ReduceOp,
          typename _TransformOp, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                            _InitType __init, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    assert(__n > 0);

    // Get the work group size adjusted to the local memory limit.
    // Pessimistically double the memory requirement to take into account memory used by compiled kernel.
    // TODO: find a way to generalize getting of reliable work-group size.
    ::std::size_t __work_group_size = oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, sizeof(_Tp) * 2);

    // Limit work-group size to 256 for performance on GPUs. Empirically tested.
    __work_group_size = ::std::max(__work_group_size, (::std::size_t)256);

    // Enable 4-wide vectorization and limit to 32 for performance on GPUs. Empirically tested.
    ::std::size_t __iters_per_work_item = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size);
    __adjust_iters_per_work_item<_VecSize>(__iters_per_work_item);

    // Use single work group implementation.
    if (__iters_per_work_item <= 32)
    {
        return __parallel_transform_reduce_small_impl<_Tp, _Commutative, _VecSize>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __work_group_size, __iters_per_work_item, __reduce_op,
            __transform_op, __init, ::std::forward<_Ranges>(__rngs)...);
    }
    // Use two-step tree reduction.
    // First step: Multiple work groups reduces (multiple) tiles of __work_group_size * __iters_per_work_item_device_kernel
    // elements. Each work group stores its local result in shared memory.
    // Second step: A single work group reduces the partial results from the first step (up to __work_group_size *
    // __iters_per_work_item_work_group_kernel elements).
    else
    {
        ::std::size_t __iters_per_work_item_device_kernel =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, 32 * __work_group_size);
        __adjust_iters_per_work_item<_VecSize>(__iters_per_work_item_device_kernel);
        ::std::size_t __iters_per_work_item_work_group_kernel = 1;
        if (__iters_per_work_item_device_kernel > 32)
        {
            __iters_per_work_item_work_group_kernel =
                oneapi::dpl::__internal::__dpl_ceiling_div(__iters_per_work_item_device_kernel, 32);
            __adjust_iters_per_work_item<_VecSize>(__iters_per_work_item_work_group_kernel);
            __iters_per_work_item_device_kernel = 32;
        }
        return __parallel_transform_reduce_impl<_Tp, _Commutative, _VecSize>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __work_group_size, __iters_per_work_item_device_kernel,
            __iters_per_work_item_work_group_kernel, __reduce_op, __transform_op, __init,
            ::std::forward<_Ranges>(__rngs)...);
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
