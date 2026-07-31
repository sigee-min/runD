#pragma once

#include "../cases.hpp"

#include "contract/support.hpp"
#include "test/assert.hpp"

#include <kernel/program/skeleton/callback.hpp>
#include <kernel/program/skeleton/shape.hpp>
#include <kernel/program/skeleton/model.hpp>
#include <kernel/program/skeleton/view/access.hpp>
#include <kernel/program/skeleton/view/factory.hpp>
#include <kernel/program/skeleton/view/linear.hpp>
#include <kernel/program/skeleton/view/type.hpp>
#include <kernel/program/skeleton/view/validation.hpp>
#include <kernel/program/skeleton/view.hpp>
#include <kernel/program/executor.hpp>
#include <kernel/program/report.hpp>
#include <kernel/program/skeleton.hpp>
#include <math32/math32.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace program_skeleton_contract {

void FunctionPointerCallback(rund::kernel::Index<1u>) noexcept;

struct VirtualCallback {
  virtual void operator()(rund::kernel::Index<1u>) noexcept {}
};

struct RvalueOnlyIndexCallback {
  rund::kernel::u64* sum = nullptr;

  void operator()(rund::kernel::Index<2u>&& index) const noexcept {
    *sum += (index[0u] * 10u) + index[1u];
  }
};

struct OverloadSensitiveIndexCallback {
  bool* lvalue_called = nullptr;
  bool* rvalue_called = nullptr;

  void operator()(const rund::kernel::Index<2u>&) const noexcept {
    *lvalue_called = true;
  }

  void operator()(rund::kernel::Index<2u>&&) const noexcept {
    *rvalue_called = true;
  }
};

struct FoldOverloadSensitiveIndexCallback {
  bool* lvalue_called = nullptr;
  bool* rvalue_called = nullptr;

  rund::kernel::u64 operator()(rund::kernel::u64 accumulator,
                               const rund::kernel::Index<2u>&) const noexcept {
    *lvalue_called = true;
    return accumulator;
  }

  rund::kernel::u64 operator()(rund::kernel::u64 accumulator,
                               rund::kernel::Index<2u>&& index) const noexcept {
    *rvalue_called = true;
    return accumulator + (index[0u] * 10u) + index[1u];
  }
};

struct NoopRank1Callback {
  void operator()(rund::kernel::Index<1u>) const noexcept {}
};

template <std::size_t Rank>
concept HasSpace = requires { typename rund::kernel::Space<Rank>; };

template <std::size_t Rank>
concept HasI32View = requires { typename rund::kernel::View<rund::kernel::i32, Rank>; };

struct TestParallelRuntimeContext {
  rund::kernel::Workspace workspace{};
  kernel_contract_test::FakePool pool{};
};

int ExpectReason(const char* expected, rund::kernel::SkeletonResult result);
rund::kernel::ParallelRuntime AcquireTestParallelRuntime(void* raw,
                                                         rund::kernel::u32 workers);
int RunSkeletonShapeContract();
int RunSkeletonValidationContract();
int RunSkeletonCapacityContract();

} // namespace program_skeleton_contract
