#pragma once

#include "src/accel/kernel/residency/device_vsm.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/window/plan.hpp>

#include <array>
#include <cstdint>
#include <memory>

namespace rund_node_test_pipeline_residency::device_vsm_test {

namespace accel = rund::node::accel::detail;

struct ProofOwner final {
  rund::kernel::LoweringArtifact artifact{};
  std::array<rund::kernel::ComputeDispatchWindow, 1u> windows{};
  std::array<std::byte, accel::DeviceVsmWindowParameterBytes> parameters{};
};

struct FakeState final {
  accel::DeviceVsmCapability capability{};
  std::uint64_t submit_count{};
  std::uint64_t final_count{};
  rund::AccelCheck rearm_check{true, "ok"};
  bool rearm_mutated{true};
};

struct Wait final {
  accel::DeviceVsmRequest request{};
  accel::DeviceVsmSubmissionControl submission_control{};
  accel::DeviceVsmFinal final{};
  std::uint64_t final_count{};
  bool valid{};
};

[[nodiscard]] accel::DeviceVsmCapability Capability() noexcept;
[[nodiscard]] rund::kernel::LoweringArtifact
CanonicalArtifact(rund::kernel::ComputeApi api);
[[nodiscard]] std::shared_ptr<accel::DeviceVsmProof> Proof(std::uint64_t pages);
void BuildRequest(Wait &, const std::shared_ptr<FakeState> &,
                  std::uint64_t pages);

[[nodiscard]] rund::AccelCheck
FakeSubmit(const accel::DeviceVsmRequest &) noexcept;
[[nodiscard]] accel::DeviceVsmRearmResult
FakeRearm(const std::shared_ptr<void> &,
          const std::shared_ptr<const accel::DeviceVsmProof> &) noexcept;

[[nodiscard]] bool CheckGeometry() noexcept;
[[nodiscard]] bool CheckProductEvidence() noexcept;
[[nodiscard]] bool CheckSurface() noexcept;
[[nodiscard]] bool CheckAuthority();
[[nodiscard]] bool CheckSource();
[[nodiscard]] bool CheckGraphPointwiseSource();
[[nodiscard]] bool CheckScanSource();
[[nodiscard]] bool CheckGraphMapScanSource();
[[nodiscard]] bool CheckGraphTerminal();
[[nodiscard]] bool CheckReduceSource();
[[nodiscard]] bool CheckReduceTerminal();

} // namespace rund_node_test_pipeline_residency::device_vsm_test
