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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"

#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)
#include _PSTL_TEST_HEADER(functional)

#include <functional>
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"

template <sycl::usm::alloc alloc_type, typename KernelName>
void
test_with_usm()
{
    sycl::queue q = TestUtils::get_test_queue();

    constexpr int n = 9;

    //data initialization
    int keys1 [n] = { 11, 11, 21, 20, 21, 21, 21, 37, 37 };
    int keys2 [n] = { 11, 11, 20, 20, 20, 21, 21, 37, 37 };
    int values[n] = {  0,  1,  2,  3,  4,  5,  6,  7,  8 };
    int output_values[n] = { };

    // allocate USM memory and copying data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper1(q, keys1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper2(q, keys2, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper3(q, values, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper4(q, output_values, n);
    auto d_keys1         = dt_helper1.get_data();
    auto d_keys2         = dt_helper2.get_data();
    auto d_values        = dt_helper3.get_data();
    auto d_output_values = dt_helper4.get_data();

    //make zip iterators
    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1, d_keys2);
    auto end_keys_in   = oneapi::dpl::make_zip_iterator(d_keys1 + n, d_keys2 + n);

    //run exclusive_scan_by_segment algorithm 
    oneapi::dpl::exclusive_scan_by_segment(
        TestUtils::make_device_policy<KernelName>(q), begin_keys_in,
        end_keys_in, d_values, d_output_values, 1,
        ::std::equal_to<>(), ::std::plus<>());

    //retrieve result on the host and check the result
    dt_helper4.retrieve_data(output_values);

//Dump
#if 0
    for(int i=0; i < n; i++) {
        std::cout << "{" << output_values1[i] << ", " << output_values2[i] << "}" << std::endl;
    }
#endif

    // Expected output
    // {11, 11}: 1
    // {11, 11}: 1
    // {21, 20}: 1
    // {20, 20}: 1
    // {21, 20}: 1
    // {21, 21}: 1
    // {21, 21}: 6
    // {37, 37}: 1
    // {37, 37}: 8
    const int exp_values[n] = {1, 1, 1, 1, 1, 1, 6, 1, 8};
    EXPECT_EQ_N(exp_values, output_values, n, "wrong values from exclusive_scan_by_segment");
}
#endif

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    // Run tests for USM shared memory
    test_with_usm<sycl::usm::alloc::shared, class KernelName1>();
    // Run tests for USM device memory
    test_with_usm<sycl::usm::alloc::device, class KernelName2>();
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
