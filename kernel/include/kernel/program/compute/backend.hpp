#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

struct ComputeDispatchWindow {
  u64 begin_sequence = 0u;
  u64 tile_count = 0u;
};

struct ComputeBackendDispatch {
  void* context = nullptr;
  bool (*execute)(void* context,
                  const ComputePlan& plan,
                  const LoweringArtifact& artifact,
                  const ComputeDispatchWindow* windows,
                  u64 window_count,
                  const BindingSet& bindings) = nullptr;
  const char* (*last_error)(void* context) = nullptr;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return context != nullptr && execute != nullptr;
  }
};

}  // namespace rund::kernel
