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
class __reduce_kernel;

//------------------------------------------------------------------------
// parallel_transform_reduce - async patterns
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------

// Parallel_transform_reduce for a small arrays using a single work group.
// Transforms and reduces __work_group_size * __iters_per_work_item elements.
template <typename _Tp, ::std::uint16_t __work_group_size, typename _KernelName>
struct __parallel_transform_reduce_small_submitter;

template <typename _Tp, ::std::uint16_t __work_group_size, typename... _Name>
struct __parallel_transform_reduce_small_submitter<_Tp, __work_group_size, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, const _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op,
               _InitType __init, _Ranges&&... __rngs) const
    {
        const ::std::uint16_t __iters_per_work_item =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size);
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _TransformOp>{__reduce_op, __transform_op};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        const _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

        __usm_host_or_buffer_storage<_ExecutionPolicy, _Tp> __res_container(__exec, 1);

        sycl::event __reduce_event =
            __exec.queue().submit([&, __n, __iters_per_work_item, __n_items](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
                auto __res_acc = __res_container.__get_acc(__cgh);
                __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
                __cgh.parallel_for<_Name...>(
                    sycl::nd_range<1>(sycl::range<1>(__work_group_size), sycl::range<1>(__work_group_size)),
                    [=](sycl::nd_item<1> __item_id) {
                        const auto __res_ptr = __res_acc.__get_pointer();
                        const auto __local_idx = __item_id.get_local_id(0);
                        // 1. Initialization (transform part). Fill local memory
                        __transform_pattern(__iters_per_work_item, __item_id, __n, __temp_local, __rngs...);
                        __dpl_sycl::__group_barrier(__item_id);
                        // 2. Reduce within work group using local memory
                        _Tp __result =
                            __reduce_pattern(unseq_backend::_RedType::__final_red, __item_id, __n_items, __temp_local);
                        if (__local_idx == 0)
                        {
                            __reduce_pattern.apply_init(__init, __result);
                            __res_ptr[0] = __result;
                        }
                    });
            });

        return __future(__reduce_event, __res_container);
    }
}; // struct __parallel_transform_reduce_small_submitter

template <typename _Tp, ::std::uint16_t __work_group_size, typename _ExecutionPolicy, typename _Size,
          typename _ReduceOp, typename _TransformOp, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_small_impl(_ExecutionPolicy&& __exec, const _Size __n, _ReduceOp __reduce_op,
                                       _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<_CustomName>;

    return __parallel_transform_reduce_small_submitter<_Tp, __work_group_size, _ReduceKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init,
        ::std::forward<_Ranges>(__rngs)...);
}

// General implementation using multiple work groups. Partial results are combined by the last work group to finish
// their partial reduction. Completion is signaled with an global atomic counter.
template <typename _Tp, typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp,
          typename _InitType, oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
          typename... _Ranges>
auto
__parallel_transform_reduce_impl(_ExecutionPolicy&& __exec, const _Size __n, ::std::size_t __work_group_size,
                                 _ReduceOp __reduce_op, _TransformOp __transform_op, _InitType __init,
                                 _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _ReduceKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<__reduce_kernel, _CustomName, _ReduceOp,
                                                                               _TransformOp, _Ranges...>;
    auto __transform_pattern =
        unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _TransformOp>{__reduce_op, __transform_op};
    auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

    ::std::size_t __max_compute_units = oneapi::dpl::__internal::__max_compute_units(__exec);
    ::std::size_t __iters_per_work_item = __n / (__max_compute_units * __work_group_size);

#if _ONEDPL_COMPILE_KERNEL
    auto __kernel = __internal::__kernel_compiler<_ReduceKernel>::__compile(__exec);
    __work_group_size = ::std::min(__work_group_size,
                                   (::std::size_t)oneapi::dpl::__internal::__kernel_work_group_size(__exec, __kernel));
#endif

    __iters_per_work_item = std::max(::std::size_t(1), __iters_per_work_item);
    // number of buffer elements processed within workgroup
    _Size __size_per_work_group = __iters_per_work_item * __work_group_size;
    _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
    _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

    // Create temporary global buffers to store temporary values
    sycl::buffer<_Tp> __temp{sycl::range<1>(__n_groups)};
    sycl::buffer<int> __done_counter(sycl::range<1>(1));
    __usm_host_or_buffer_storage<_ExecutionPolicy, _Tp> __res_container(__exec, 1);

    sycl::event __reduce_event = __exec.queue().submit([&, __n, __n_groups, __n_items](sycl::handler& __cgh) {
        oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
        sycl::accessor __temp_acc{__temp, __cgh, sycl::read_write, __dpl_sycl::__no_init{}};
        sycl::accessor __done_acc{__done_counter, __cgh, sycl::read_write, __dpl_sycl::__no_init{}};
        auto __res_acc = __res_container.__get_acc(__cgh);
        __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
        __dpl_sycl::__local_accessor<bool, 0> __done_local(__cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
        __cgh.use_kernel_bundle(__kernel.get_kernel_bundle());
#endif
        __cgh.parallel_for<_ReduceKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
            __kernel,
#endif
            sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size), sycl::range<1>(__work_group_size)),
            [=](sycl::nd_item<1> __item_id) {
                auto __res_ptr = __res_acc.__get_pointer();
                const auto __local_idx = __item_id.get_local_id(0);
                const auto __group_idx = __item_id.get_group(0);

                // 0. Init done counters
                if (__local_idx == 0)
                {
                    __done_local = false;
                    if (__group_idx == 0)
                        __done_acc[0] = 0;
                }

                // 1. Initialization (transform part). Fill local memory
                __transform_pattern(__iters_per_work_item, __item_id, __n, __temp_local, __rngs...);
                __dpl_sycl::__group_barrier(__item_id);

                // 2. Reduce within work group using local memory
                _Tp __tmp_result =
                    __reduce_pattern(unseq_backend::_RedType::__partial_red, __item_id, __n_items, __temp_local);

                // 3. Increment done counter
                if (__local_idx == 0)
                {
                    __temp_acc[__group_idx] = __tmp_result;
                    __dpl_sycl::__atomic_ref<int, sycl::access::address_space::global_space> __done_atomic(
                        __done_acc[0]);
                    int __done = __done_atomic++;
                    if (__done + 1 == __n_groups)
                    {
                        __done_local = true;
                        __done_atomic = 0;
                    }
                }
                __dpl_sycl::__group_barrier(__item_id);

                // 4. Reduce across work groups
                if (__done_local)
                {
                    _Tp __result =
                        __reduce_pattern(unseq_backend::_RedType::__final_red, __item_id, __n_groups, __temp_acc);
                    if (__local_idx == 0)
                    {
                        __reduce_pattern.apply_init(__init, __result);
                        __res_ptr[0] = __result;
                    }
                }
            });
    });

    return __future(__reduce_event, __res_container);
}

// General version of parallel_transform_reduce.
// The binary operator must be associative but commutativity is only required by some of the algorithms using
// __parallel_transform_reduce. This is provided by the _Commutative parameter. The current implementation uses a
// generic implementation that processes elements in order. However, future improvements might be possible utilizing
// the commutative property of the respective algorithms.
//
// Each work item transforms and reduces __iters_per_work_item elements from global memory and stores the result in SLM.
// Each work group of size __work_group_size reduces the preliminary results of each work item in a group reduction
// using SLM.
// A single-work group implementation is used for small arrays.
// Larger arrays use a general implementation utilizing multiple work groups.
template <typename _Tp, typename _Commutative, typename _ExecutionPolicy, typename _ReduceOp, typename _TransformOp,
          typename _InitType, oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
          typename... _Ranges>
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

    // Use single work group implementation if array < __work_group_size * __iters_per_work_item.
    if (__work_group_size >= 256 && __n <= 8192)
    {
        return __parallel_transform_reduce_small_impl<_Tp, 256>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                __reduce_op, __transform_op, __init,
                                                                ::std::forward<_Ranges>(__rngs)...);
    }
    // Otherwise use multi work-group reduction using a global atomic counter to reduce the partial results.
    return __parallel_transform_reduce_impl<_Tp>(::std::forward<_ExecutionPolicy>(__exec), __n, __work_group_size,
                                                 __reduce_op, __transform_op, __init,
                                                 ::std::forward<_Ranges>(__rngs)...);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
