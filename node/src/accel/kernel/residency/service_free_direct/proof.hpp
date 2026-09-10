#pragma once

#include "identity.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/limit.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

enum class ServiceFreeDirectRetention : std::uint8_t { Terminal, History };

// Fixed-size projection of one resident recurrence. `semantic_owner` retains
// the exact RunState that owns artifact/window/parameter storage; normalized
// resident rows and their independent storage handles are copied by value.
// Therefore no proof-owned storage scales with the iteration count.
struct ServiceFreeDirectProof final {
  static constexpr std::size_t BindingCapacity =
      static_cast<std::size_t>(rund::kernel::kMaxComputeBindingCount);
  static constexpr std::size_t StateCapacity = BindingCapacity * 2u;

  ServiceFreeDirectIdentity identity{};
  std::shared_ptr<const void> semantic_owner{};
  const rund::kernel::LoweringArtifact *artifact{};
  rund::kernel::ComputePlan plan{};
  std::array<rund::kernel::ResidentBufferRef, BindingCapacity> inputs{};
  std::array<std::shared_ptr<void>, BindingCapacity> input_handles{};
  std::array<rund::kernel::ResidentBufferRef, BindingCapacity> outputs{};
  std::array<std::shared_ptr<void>, BindingCapacity> output_handles{};
  std::array<std::uint64_t, BindingCapacity> output_pitch_bytes{};
  // Canonical physical-storage rows touched anywhere in the recurrence. A
  // row spans the complete allocation (offset zero, byte stride one) while
  // `inputs` and `outputs` retain the exact semantic views. Access is the
  // union of every occurrence. The fixed capacity is part of eligibility:
  // a recurrence that introduces Q-distinct storage cannot claim fixed
  // common state merely because its backend command stream is fused.
  std::array<rund::kernel::ResidentBufferRef, StateCapacity> states{};
  std::array<std::shared_ptr<void>, StateCapacity> state_handles{};
  const rund::kernel::ComputeDispatchWindow *windows{};
  const std::byte *parameters{};
  std::uint64_t iterations{};
  std::uint64_t window_count{};
  std::uint64_t parameter_bytes{};
  std::uint32_t input_count{};
  std::uint32_t output_count{};
  std::uint32_t state_count{};
  // The common owner retains only the fixed proof, one semantic RunState,
  // and fixed-capacity status/control rows after backend preparation. Cold
  // authored occurrences may affect transient preparation cost but are not
  // retained by the admitted product request.
  bool fixed_common_storage{};
  ServiceFreeDirectRetention retention{ServiceFreeDirectRetention::Terminal};
};

} // namespace rund::node::accel::detail
