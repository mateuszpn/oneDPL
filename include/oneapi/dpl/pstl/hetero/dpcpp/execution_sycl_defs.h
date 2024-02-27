// -*- C++ -*-
//===-- execution_sycl_defs.h ---------------------------------------------===//
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

#ifndef _ONEDPL_EXECUTION_SYCL_DEFS_H
#define _ONEDPL_EXECUTION_SYCL_DEFS_H

#include "../../onedpl_config.h"
#include "../../execution_defs.h"

#include "sycl_defs.h"

#include <type_traits>
#include <mutex>
#include <optional>

namespace oneapi
{
namespace dpl
{
namespace execution
{
inline namespace __dpl
{

struct DefaultKernelName;

////////////////////////////////////////////////////////////////////////////////
// sycl_queue_container - container with optionally created sycl::queue instance
template <class TSYCLQueueFactory>
class sycl_queue_container
{
  public:
    template <typename... Args>
    sycl::queue
    get_queue(Args&&... args)
    {
        ::std::call_once(__is_created, [&]() {
            TSYCLQueueFactory factory;
            __queue.emplace(factory(::std::forward<Args>(args)...));
        });

        assert(__queue.has_value());
        return __queue.value();
    }

  private:
    ::std::once_flag __is_created;
    ::std::optional<sycl::queue> __queue;
};
template <typename TFactory>
using sycl_queue_container_ptr = ::std::shared_ptr<sycl_queue_container<TFactory>>;

////////////////////////////////////////////////////////////////////////////////
// sycl_queue_factory_device - default sycl::queue factory for device policy
struct sycl_queue_factory_device
{
    template <typename... Args>
    sycl::queue
    operator()(Args&&... args)
    {
        return sycl::queue(::std::forward<Args>(args)...);
    }
};

//We can create device_policy object:
// 1. from sycl::queue
// 2. from sycl::device_selector (implicitly through sycl::queue)
// 3. from sycl::device
// 4. from other device_policy encapsulating the same queue type
template <typename KernelName = DefaultKernelName, class TSyclQueueFactory = sycl_queue_factory_device>
class device_policy
{
  public:
    using kernel_name = KernelName;

    device_policy() : q_container(::std::make_shared<sycl_queue_container<TSyclQueueFactory>>()) {}

    template <typename OtherName>
    device_policy(const device_policy<OtherName, TSyclQueueFactory>& other)
        : q_container(other.get_sycl_queue_container())
    {
    }
    explicit device_policy(sycl::queue q_) : device_policy() { q_container->get_queue(::std::move(q_)); }
    explicit device_policy(sycl::device d_) : device_policy() { q_container->get_queue(::std::move(d_)); }
    operator sycl::queue() const { return queue(); }
    sycl::queue
    queue() const
    {
        return q_container->get_queue();
    }

    auto
    get_sycl_queue_container() const
    {
        return q_container;
    }

    // For internal use only
    static constexpr ::std::true_type
    __allow_unsequenced()
    {
        return ::std::true_type{};
    }
    // __allow_vector is needed for __is_vectorization_preferred
    static constexpr ::std::true_type
    __allow_vector()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::true_type
    __allow_parallel()
    {
        return ::std::true_type{};
    }

  private:
    mutable sycl_queue_container_ptr<TSyclQueueFactory> q_container;
};

#if _ONEDPL_FPGA_DEVICE

////////////////////////////////////////////////////////////////////////////////
// sycl_queue_factory_fpga - default sycl::queue FPGA factory
struct sycl_queue_factory_fpga
{
    template <class... Args>
    sycl::queue
    operator()(Args&&... args)
    {
        if constexpr (sizeof...(Args) > 0)
        {
            return sycl::queue(::std::forward<Args>(args)...);
        }
        else
        {
            return sycl::queue(
#    if _ONEDPL_FPGA_EMU
                __dpl_sycl::__fpga_emulator_selector()
#    else
                __dpl_sycl::__fpga_selector()
#    endif // _ONEDPL_FPGA_EMU
            );
        }
    }
};

struct DefaultKernelNameFPGA;
template <unsigned int factor = 1, typename KernelName = DefaultKernelNameFPGA,
          class TSYCLQueueFactory = sycl_queue_factory_fpga>
class fpga_policy : public device_policy<KernelName, TSYCLQueueFactory>
{
    using base = device_policy<KernelName, TSYCLQueueFactory>;

  public:
    static constexpr unsigned int unroll_factor = factor;

    fpga_policy() = default;

    template <unsigned int other_factor, typename OtherName>
    fpga_policy(const fpga_policy<other_factor, OtherName, TSYCLQueueFactory>& other) : base(other){};
    explicit fpga_policy(sycl::queue q) : base(q) {}
    explicit fpga_policy(sycl::device d) : base(d) {}

    // For internal use only

    const base&
    __device_policy() const
    {
        return static_cast<const base&>(*this);
    };
};

#endif // _ONEDPL_FPGA_DEVICE

// 2.8, Execution policy objects
#if _ONEDPL_PREDEFINED_POLICIES

// In order to be useful oneapi::dpl::execution::dpcpp_default.queue() from one translation unit should be equal to
// oneapi::dpl::execution::dpcpp_default.queue() from another TU.
// Starting with c++17 we can simply define sycl as inline variable.
#    if _ONEDPL___cplusplus >= 201703L

inline device_policy<> dpcpp_default{};
#        if _ONEDPL_FPGA_DEVICE
inline fpga_policy<> dpcpp_fpga{};
#        endif // _ONEDPL_FPGA_DEVICE

#    endif // _ONEDPL___cplusplus >= 201703L

#endif // _ONEDPL_PREDEFINED_POLICIES

// make_policy functions
template <typename KernelName = DefaultKernelName>
device_policy<KernelName>
make_device_policy(sycl::queue q)
{
    return device_policy<KernelName>(q);
}

template <typename KernelName = DefaultKernelName>
device_policy<KernelName>
make_device_policy(sycl::device d)
{
    return device_policy<KernelName>(d);
}

template <typename NewKernelName, typename OldKernelName = DefaultKernelName>
device_policy<NewKernelName>
make_device_policy(const device_policy<OldKernelName>& policy
#if _ONEDPL_PREDEFINED_POLICIES
                   = dpcpp_default
#endif // _ONEDPL_PREDEFINED_POLICIES
)
{
    return device_policy<NewKernelName>(policy);
}

template <typename NewKernelName, typename OldKernelName = DefaultKernelName>
device_policy<NewKernelName>
make_hetero_policy(const device_policy<OldKernelName>& policy)
{
    return device_policy<NewKernelName>(policy);
}

#if _ONEDPL_FPGA_DEVICE
template <unsigned int unroll_factor = 1, typename KernelName = DefaultKernelNameFPGA>
fpga_policy<unroll_factor, KernelName>
make_fpga_policy(sycl::queue q)
{
    return fpga_policy<unroll_factor, KernelName>(q);
}

template <unsigned int unroll_factor = 1, typename KernelName = DefaultKernelNameFPGA>
fpga_policy<unroll_factor, KernelName>
make_fpga_policy(sycl::device d)
{
    return fpga_policy<unroll_factor, KernelName>(d);
}

template <unsigned int new_unroll_factor, typename NewKernelName, unsigned int old_unroll_factor = 1,
          typename OldKernelName = DefaultKernelNameFPGA>
fpga_policy<new_unroll_factor, NewKernelName>
make_fpga_policy(const fpga_policy<old_unroll_factor, OldKernelName>& policy
#    if _ONEDPL_PREDEFINED_POLICIES
                 = dpcpp_fpga
#    endif // _ONEDPL_PREDEFINED_POLICIES
)
{
    return fpga_policy<new_unroll_factor, NewKernelName>(policy);
}

template <unsigned int new_unroll_factor, typename NewKernelName, unsigned int old_unroll_factor = 1,
          typename OldKernelName = DefaultKernelNameFPGA>
fpga_policy<new_unroll_factor, NewKernelName>
make_hetero_policy(const fpga_policy<old_unroll_factor, OldKernelName>& policy)
{
    return fpga_policy<new_unroll_factor, NewKernelName>(policy);
}
#endif // _ONEDPL_FPGA_DEVICE

} // namespace __dpl

inline namespace v1
{

// 2.3, Execution policy type trait
template <typename... PolicyParams>
struct is_execution_policy<device_policy<PolicyParams...>> : ::std::true_type
{
};

#if _ONEDPL_FPGA_DEVICE
template <unsigned int unroll_factor, typename... PolicyParams>
struct is_execution_policy<fpga_policy<unroll_factor, PolicyParams...>> : ::std::true_type
{
};
#endif

} // namespace v1
} // namespace execution

namespace __internal
{
template <typename Policy>
using __policy_kernel_name = typename ::std::decay_t<Policy>::kernel_name;

template <typename Policy>
inline constexpr unsigned int __policy_unroll_factor = ::std::decay_t<Policy>::unroll_factor;

template <typename _T>
struct __is_device_execution_policy : ::std::false_type
{
};

template <typename... PolicyParams>
struct __is_device_execution_policy<execution::device_policy<PolicyParams...>> : ::std::true_type
{
};

template <typename _T>
inline constexpr bool __is_device_execution_policy_v = __is_device_execution_policy<_T>::value;

template <typename _T>
struct __is_fpga_execution_policy : ::std::false_type
{
};

#if _ONEDPL_FPGA_DEVICE
template <unsigned int unroll_factor, typename... PolicyParams>
struct __is_fpga_execution_policy<execution::fpga_policy<unroll_factor, PolicyParams...>> : ::std::true_type
{
};

template <typename _T, unsigned int unroll_factor, typename... PolicyParams>
struct __ref_or_copy_impl<execution::fpga_policy<unroll_factor, PolicyParams...>, _T>
{
    using type = _T;
};
#endif

template <typename _T, typename... PolicyParams>
struct __ref_or_copy_impl<execution::device_policy<PolicyParams...>, _T>
{
    using type = _T;
};

// Extension: hetero execution policy type trait
template <typename _T>
using __is_hetero_execution_policy =
    ::std::disjunction<__is_device_execution_policy<_T>, __is_fpga_execution_policy<_T>>;

template <typename _T>
inline constexpr bool __is_hetero_execution_policy_v = __is_hetero_execution_policy<_T>::value;

// Extension: check if parameter pack is convertible to events
template <class... _Ts>
inline constexpr bool __is_convertible_to_event = (::std::is_convertible_v<::std::decay_t<_Ts>, sycl::event> && ...);

template <typename _T, typename... _Ts>
using __enable_if_convertible_to_events = ::std::enable_if_t<__is_convertible_to_event<_Ts...>, _T>;

// Extension: execution policies type traits
template <typename _ExecPolicy, typename _T, typename... _Events>
using __enable_if_device_execution_policy = ::std::enable_if_t<
    __is_device_execution_policy_v<::std::decay_t<_ExecPolicy>> && __is_convertible_to_event<_Events...>, _T>;

template <typename _ExecPolicy, typename _T = void>
using __enable_if_hetero_execution_policy =
    ::std::enable_if_t<__is_hetero_execution_policy_v<::std::decay_t<_ExecPolicy>>, _T>;

template <typename _ExecPolicy, typename _T = void>
using __enable_if_fpga_execution_policy =
    ::std::enable_if_t<__is_fpga_execution_policy<::std::decay_t<_ExecPolicy>>::value, _T>;

template <typename _ExecPolicy, typename _T, typename _Op1, typename... _Events>
using __enable_if_device_execution_policy_single_no_default =
    ::std::enable_if_t<__is_device_execution_policy_v<::std::decay_t<_ExecPolicy>> &&
                           !::std::is_convertible_v<_Op1, sycl::event> && __is_convertible_to_event<_Events...>,
                       _T>;

template <typename _ExecPolicy, typename _T, typename _Op1, typename _Op2, typename... _Events>
using __enable_if_device_execution_policy_double_no_default =
    ::std::enable_if_t<__is_device_execution_policy_v<::std::decay_t<_ExecPolicy>> &&
                           !::std::is_convertible_v<_Op1, sycl::event> && !::std::is_convertible_v<_Op2, sycl::event> &&
                           __is_convertible_to_event<_Events...>,
                       _T>;

} // namespace __internal

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXECUTION_SYCL_DEFS_H
