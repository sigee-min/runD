#pragma once

#include "identity.hpp"
#include "../../profile.hpp"

#include <accel/check.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

enum class ServiceFreeDirectTerminal : std::uint8_t { Known, UnknownMayWrite };

struct ServiceFreeDirectEvidence final {
  ServiceFreeDirectIdentity proof{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t nonce{};
  std::uint64_t iterations{};
  std::uint64_t completed_iterations{};
  std::uint64_t native_submit_count{};
  std::uint64_t epoch_native_submit_count{};
  std::uint64_t payload_dispatch_count{};
  std::uint64_t host_service_turn_count{};
  std::uint64_t host_epoch_callback_count{};
  std::uint64_t final_callback_count{};
  std::uint64_t completed_ns{};
  bool may_write{};
};

struct ServiceFreeDirectFinal final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  ServiceFreeDirectTerminal terminal{ServiceFreeDirectTerminal::Known};
  ServiceFreeDirectEvidence evidence{};
  PreparedPipelineProfileEvidence profile{};
  std::uint32_t active_step_count{};
  std::shared_ptr<const void> profile_owner{};
};

using ServiceFreeDirectFinalCompletion =
    void (*)(void *, ServiceFreeDirectFinal &&) noexcept;

} // namespace rund::node::accel::detail
