#pragma once

#include "../callback.hpp"
#include "../pipeline.hpp"
#include "evidence.hpp"
#include "../../residency/persistent_sliding.hpp"
#include "../../residency/schedule.hpp"
#include "../../residency/sliding.hpp"
#include "../../residency/window.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

namespace rund::node::accel::detail {

struct PreparedResidencyWindowBatch final {
  PreparedKernelPipeline pipeline{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint64_t epoch{};
  std::uint32_t control_generation{};
  std::uint8_t bank{};
};

struct PreparedResidencyWindowRelease final {
  BackendResidencyWindowReceipt receipt{};
  PreparedPipelineEvidence evidence{};
};

struct PreparedResidencyWindowFinal final {
  BackendResidencyWindowFinal evidence{};
};

struct PreparedResidencyScheduleRole final {
  PreparedKernelPipeline pipeline{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint32_t first_control_generation{};
  std::uint32_t control_generation_stride{};
  std::uint8_t role{};
  std::uint8_t bank{};
};

struct PreparedResidencyScheduleRelease final {
  BackendResidencyWindowReceipt receipt{};
  PreparedPipelineEvidence evidence{};
};

struct PreparedResidencyScheduleFinal final {
  BackendResidencyScheduleFinal evidence{};
};

struct PreparedResidencySlidingRole final {
  PreparedKernelPipeline pipeline{};
  std::uint32_t first_control_generation{};
  std::uint32_t control_generation_stride{};
  std::uint8_t slot{};
};

struct PreparedResidencyPersistentSlidingRole final {
  PreparedKernelPipeline pipeline{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint32_t first_control_generation{};
  std::uint32_t control_generation_stride{};
  std::uint64_t first_descriptor_generation{};
  std::uint64_t descriptor_generation_stride{};
  std::uint8_t slot{};
};

struct PreparedResidencyPersistentSlidingPreparation final {
  PersistentResidencySlidingPreparation backend{};

  [[nodiscard]] PersistentResidencySlidingRequest
  active_request() const noexcept {
    return backend.active_request();
  }
  [[nodiscard]] explicit operator bool() const noexcept {
    if (!static_cast<bool>(backend)) {
      return false;
    }
    auto request = backend.active_request();
    request.lowering = backend.lowering;
    return persistent_sliding_request_valid(backend.capability, request);
  }
};

enum class PreparedResidencySlidingProjection : std::uint8_t {
  Ready,
  Pending,
  Failed,
};

struct PreparedResidencySlidingSelection final {
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint64_t read_mask{};
  std::uint64_t write_mask{};
  std::uint64_t descriptor_generation{};
  std::uint32_t control_generation{};
};

struct PreparedResidencySlidingRelease final {
  BackendResidencySlidingTerminal terminal{};
  PreparedPipelineEvidence evidence{};
};

using PreparedResidencySlidingProject = PreparedResidencySlidingProjection (*)(
    void *, std::uint64_t coordinate, std::uint64_t turn, std::uint8_t slot,
    PreparedResidencySlidingSelection &) noexcept;
using PreparedResidencySlidingReleaseCompletion =
    void (*)(void *, PreparedResidencySlidingRelease &&) noexcept;
using PreparedResidencySlidingReleaseReturned =
    bool (*)(void *, std::uint64_t coordinate, std::uint64_t turn,
             std::uint8_t slot) noexcept;
using PreparedResidencySlidingFinalCompletion =
    void (*)(void *, BackendResidencySlidingFinal &&) noexcept;

struct PreparedResidencySlidingRequest final {
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t coordinate_count{};
  std::array<PreparedResidencySlidingRole, ResidencySlidingCapacity> roles{};
  std::size_t role_count{};
  ResidencySlidingMemory memory{ResidencySlidingMemory::HostCoherent};
  PreparedResidencySlidingProject project{};
  PreparedResidencySlidingReleaseCompletion release{};
  PreparedResidencySlidingReleaseReturned returned{};
  PreparedResidencySlidingFinalCompletion final{};
  void *user{};
};

using PreparedResidencyScheduleReleaseCompletion =
    void (*)(void *, PreparedResidencyScheduleRelease &&) noexcept;
using PreparedResidencyScheduleFinalCompletion =
    void (*)(void *, PreparedResidencyScheduleFinal &&) noexcept;

struct PreparedResidencyScheduleRequest final {
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch_count{};
  std::array<PreparedResidencyScheduleRole, ResidencyScheduleRoleCapacity>
      roles{};
  std::size_t role_count{};
  std::size_t tail_local_count{};
  std::shared_ptr<void> lowering{};
  std::shared_ptr<void> admission{};
  PreparedResidencyScheduleReleaseCompletion release{};
  PreparedResidencyScheduleFinalCompletion final{};
  void *user{};
};

using PreparedResidencyWindowReleaseCompletion =
    void (*)(void *, PreparedResidencyWindowRelease &&) noexcept;
using PreparedResidencyWindowFinalCompletion =
    void (*)(void *, PreparedResidencyWindowFinal &&) noexcept;

struct PreparedResidencyWindowRequest final {
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t first_epoch{};
  std::array<PreparedResidencyWindowBatch, ResidencyWindowCapacity> batches{};
  std::size_t batch_count{};
  PreparedResidencyWindowReleaseCompletion release{};
  PreparedResidencyWindowFinalCompletion final{};
  void *user{};
};

namespace prepared {
struct PipelineState;
namespace sliding {
struct State;
}
} // namespace prepared
struct PreparedResidencyStreamControl;

struct PreparedResidencyWindowControl final {
  std::mutex gate{};
  PreparedResidencyWindowRequest bound{};
  BackendResidencyWindowRequest backend{};
  std::array<prepared::PipelineState *, ResidencyWindowCapacity> states{};
  std::array<BackendResidencyWindowReceipt, ResidencyWindowCapacity> receipts{};
  std::size_t release_count{};
  PreparedResidencyStreamControl *stream{};
  bool active{};
  bool aborting{};
  bool quarantined{};
};

struct PreparedResidencyStreamControl final {
  std::mutex gate{};
  std::array<prepared::PipelineState *, ResidencyWindowCapacity> states{};
  std::array<PreparedKernelPipeline, ResidencyWindowCapacity> owners{};
  std::size_t state_count{};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  bool active{};
  bool quarantined{};
};

struct PreparedResidencyScheduleControl final {
  std::mutex gate{};
  PreparedResidencyScheduleRequest bound{};
  BackendResidencyScheduleRequest backend{};
  std::array<prepared::PipelineState *, ResidencyScheduleRoleCapacity> states{};
  PreparedResidencyStreamControl *stream{};
  std::uint64_t release_count{};
  std::uint64_t success_prefix{};
  std::uint64_t suppressed_first{};
  std::uint64_t suppressed_count{};
  rund::AccelCheck first_failure{true, "ok"};
  bool active{};
  bool aborting{};
  bool malformed{};
  bool quarantined{};
};

struct PreparedResidencySlidingControl final {
  std::shared_ptr<prepared::sliding::State> state{};
  BackendResidencySlidingCapability capability{};
};

} // namespace rund::node::accel::detail
