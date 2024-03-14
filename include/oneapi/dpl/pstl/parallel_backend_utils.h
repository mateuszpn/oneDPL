// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_UTILS_H
#define _ONEDPL_PARALLEL_BACKEND_UTILS_H

#include <iterator>
#include <utility>
#include <cassert>
#include <memory>
#include "utils.h"

namespace oneapi
{
namespace dpl
{
namespace __utils
{

//------------------------------------------------------------------------
// raw buffer (with specified _TAllocator)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Tp, template <typename _Tp> typename _TAllocator>
class __buffer_impl
{
    struct __buffer_data
    {
        _TAllocator<_Tp> __allocator_;
        _Tp* __allocated_mem = nullptr;
        const ::std::size_t __buf_size_ = 0;

        __buffer_data(std::size_t __n) : __allocator_(), __allocated_mem(__allocator_.allocate(__n)), __buf_size_(__n)
        {
            // As described at https://en.cppreference.com/w/cpp/memory/allocator/allocate
            // the exceptions may be throwed from the std::allocator<T>::allocate call :
            //  - throws std::bad_array_new_length if std::numeric_limits<std::size_t>::max() / sizeof(T) < n.
            //  - throws std::bad_alloc if allocation fails.

            // So the initialization order shoud be exactly like here:
            // 1. Initialize allocator.
            // 2. Allocate memory.
            // 3. Save buffer size.
        }
    };
    struct __buffer_data_custom_deleter
    {
        void
        operator()(__buffer_data* pData)
        {
            if (pData != nullptr)
                pData->__allocator_.deallocate(pData->__allocated_mem, __ptr_->__buf_size_);
        }
    };
    using __data_ptr_t = ::std::unique_ptr<__buffer_data, __buffer_data_custom_deleter>;
    __data_ptr_t __ptr;

  public:
    static_assert(::std::is_same_v<_ExecutionPolicy, ::std::decay_t<_ExecutionPolicy>>);

    __buffer_impl(_ExecutionPolicy /*__exec*/, ::std::size_t __n) : __ptr(::std::make_unique(__n)) {}

    _Tp*
    get() const
    {
        return __ptr_->__allocated_mem;
    }

    operator bool() const { return get() != nullptr; }
};

//! Destroy sequence [xs,xe)
struct __serial_destroy
{
    template <typename _RandomAccessIterator>
    void
    operator()(_RandomAccessIterator __zs, _RandomAccessIterator __ze)
    {
        typedef typename ::std::iterator_traits<_RandomAccessIterator>::value_type _ValueType;
        while (__zs != __ze)
        {
            --__ze;
            (*__ze).~_ValueType();
        }
    }
};

//! Merge sequences [__xs,__xe) and [__ys,__ye) to output sequence [__zs,(__xe-__xs)+(__ye-__ys)), using ::std::move
struct __serial_move_merge
{
    const ::std::size_t _M_nmerge;

    explicit __serial_move_merge(::std::size_t __nmerge) : _M_nmerge(__nmerge) {}
    template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Compare,
              class _MoveValueX, class _MoveValueY, class _MoveSequenceX, class _MoveSequenceY>
    void
    operator()(_RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe, _RandomAccessIterator2 __ys,
               _RandomAccessIterator2 __ye, _RandomAccessIterator3 __zs, _Compare __comp, _MoveValueX __move_value_x,
               _MoveValueY __move_value_y, _MoveSequenceX __move_sequence_x, _MoveSequenceY __move_sequence_y)
    {
        constexpr bool __same_move_val = ::std::is_same_v<_MoveValueX, _MoveValueY>;
        constexpr bool __same_move_seq = ::std::is_same_v<_MoveSequenceX, _MoveSequenceY>;

        auto __n = _M_nmerge;
        assert(__n > 0);

        auto __nx = __xe - __xs;
        //auto __ny = __ye - __ys;
        _RandomAccessIterator3 __zs_beg = __zs;

        if (__xs != __xe)
        {
            if (__ys != __ye)
            {
                for (;;)
                {
                    if (__comp(*__ys, *__xs))
                    {
                        const auto __i = __zs - __zs_beg;
                        if (__i < __nx)
                            __move_value_x(__ys, __zs);
                        else
                            __move_value_y(__ys, __zs);
                        ++__zs, --__n;
                        if (++__ys == __ye)
                        {
                            break;
                        }
                        else if (__n == 0)
                        {
                            const auto __j = __zs - __zs_beg;
                            if (__same_move_seq || __j < __nx)
                                __zs = __move_sequence_x(__ys, __ye, __zs);
                            else
                                __zs = __move_sequence_y(__ys, __ye, __zs);
                            break;
                        }
                    }
                    else
                    {
                        const auto __i = __zs - __zs_beg;
                        if (__same_move_val || __i < __nx)
                            __move_value_x(__xs, __zs);
                        else
                            __move_value_y(__xs, __zs);
                        ++__zs, --__n;
                        if (++__xs == __xe)
                        {
                            const auto __j = __zs - __zs_beg;
                            if (__same_move_seq || __j < __nx)
                                __move_sequence_x(__ys, __ye, __zs);
                            else
                                __move_sequence_y(__ys, __ye, __zs);
                            return;
                        }
                        else if (__n == 0)
                        {
                            const auto __j = __zs - __zs_beg;
                            if (__same_move_seq || __j < __nx)
                            {
                                __zs = __move_sequence_x(__xs, __xe, __zs);
                                __move_sequence_x(__ys, __ye, __zs);
                            }
                            else
                            {
                                __zs = __move_sequence_y(__xs, __xe, __zs);
                                __move_sequence_y(__ys, __ye, __zs);
                            }
                            return;
                        }
                    }
                }
            }
            __ys = __xs;
            __ye = __xe;
        }
        const auto __i = __zs - __zs_beg;
        if (__same_move_seq || __i < __nx)
            __move_sequence_x(__ys, __ye, __zs);
        else
            __move_sequence_y(__ys, __ye, __zs);
    }
};

template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator, typename _Compare,
          typename _CopyConstructRange>
_OutputIterator
__set_union_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                      _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                      _CopyConstructRange __cc_range)
{
    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;

    for (; __first1 != __last1; ++__result)
    {
        if (__first2 == __last2)
            return __cc_range(__first1, __last1, __result);
        if (__comp(*__first2, *__first1))
        {
            ::new (::std::addressof(*__result)) _Tp(*__first2);
            ++__first2;
        }
        else
        {
            ::new (::std::addressof(*__result)) _Tp(*__first1);
            if (!__comp(*__first1, *__first2))
                ++__first2;
            ++__first1;
        }
    }
    return __cc_range(__first2, __last2, __result);
}

template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator, typename _Compare>
_OutputIterator
__set_intersection_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                             _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp)
{
    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;

    for (; __first1 != __last1 && __first2 != __last2;)
    {
        if (__comp(*__first1, *__first2))
            ++__first1;
        else
        {
            if (!__comp(*__first2, *__first1))
            {
                ::new (::std::addressof(*__result)) _Tp(*__first1);
                ++__result;
                ++__first1;
            }
            ++__first2;
        }
    }
    return __result;
}

template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator, typename _Compare,
          typename _CopyConstructRange>
_OutputIterator
__set_difference_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                           _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                           _CopyConstructRange __cc_range)
{
    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;

    for (; __first1 != __last1;)
    {
        if (__first2 == __last2)
            return __cc_range(__first1, __last1, __result);

        if (__comp(*__first1, *__first2))
        {
            ::new (::std::addressof(*__result)) _Tp(*__first1);
            ++__result;
            ++__first1;
        }
        else
        {
            if (!__comp(*__first2, *__first1))
                ++__first1;
            ++__first2;
        }
    }
    return __result;
}
template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator, typename _Compare,
          typename _CopyConstructRange>
_OutputIterator
__set_symmetric_difference_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                                     _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                                     _CopyConstructRange __cc_range)
{
    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;

    for (; __first1 != __last1;)
    {
        if (__first2 == __last2)
            return __cc_range(__first1, __last1, __result);

        if (__comp(*__first1, *__first2))
        {
            ::new (::std::addressof(*__result)) _Tp(*__first1);
            ++__result;
            ++__first1;
        }
        else
        {
            if (__comp(*__first2, *__first1))
            {
                ::new (::std::addressof(*__result)) _Tp(*__first2);
                ++__result;
            }
            else
                ++__first1;
            ++__first2;
        }
    }
    return __cc_range(__first2, __last2, __result);
}

} // namespace __utils
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_UTILS_H
