// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <execution>
#include <oneapi/tbb/concurrent_vector.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::concurrent_vector<int> cv;
    return TestUtils::done();
}
