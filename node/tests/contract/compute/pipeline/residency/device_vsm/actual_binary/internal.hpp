#pragma once

#include "../actual_binary.hpp"
#include "../local.hpp"
#include "../wait.hpp"

#include "src/accel/context/local.hpp"
#include "src/accel/context/transfer.hpp"
#include "src/compute/device/residency/execution/device_vsm/registration.hpp"
#include "src/compute/device/state.hpp"

#include <kernel/program/compute/artifact.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail {

namespace accel = rund::node::accel::detail;
namespace residency = rund::compute::detail::residency;

struct PreparedBinary final {
  rund::compute::Backend backend{};
  std::shared_ptr<rund::compute::detail::DeviceState> device{};
  const rund::compute::detail::AccelDeviceState *native{};
  std::uint64_t pages{};
  std::size_t elements{};
  std::vector<std::uint32_t> first{};
  std::vector<std::uint32_t> second{};
  std::vector<std::uint32_t> zero{};
  std::array<rund::AccelBuffer, 3u> buffers{};
  std::array<accel::UploadRoute, 3u> routes{};
  std::shared_ptr<accel::DeviceVsmProof> proof{};
  accel::DeviceVsmPreparation preparation{};
  std::shared_ptr<wait_detail::Owner> wait{};
};

[[nodiscard]] bool RejectBinary(rund::compute::Backend, std::uint64_t,
                                const char *) noexcept;

[[nodiscard]] rund::kernel::ComputeApi ApiFor(rund::compute::Backend) noexcept;

[[nodiscard]] rund::kernel::LoweringArtifact
    BinaryArtifact(rund::kernel::ComputeApi);

[[nodiscard]] accel::DeviceVsmResidentBinding
Binding(const accel::UploadRoute &, accel::DeviceVsmResidentRole) noexcept;

[[nodiscard]] std::shared_ptr<accel::DeviceVsmProof>
BuildProof(rund::kernel::ComputeApi, std::uint64_t,
           const std::array<accel::UploadRoute, 3u> &);

[[nodiscard]] std::unique_ptr<PreparedBinary>
PrepareBinary(rund::compute::Backend,
              const std::shared_ptr<rund::compute::detail::DeviceState> &,
              std::uint64_t, ActualPrepare, ActualQueueCount);

[[nodiscard]] wait_detail::Result
CloseAuthority(wait_detail::Owner &, const accel::DeviceVsmFinal &) noexcept;

[[nodiscard]] bool ValidateBinaryOutputAndEvidence(PreparedBinary &,
                                                   std::uint64_t &retained);

[[nodiscard]] bool
RunBinary(rund::compute::Backend,
          const std::shared_ptr<rund::compute::detail::DeviceState> &,
          std::uint64_t, ActualPrepare, ActualQueueCount,
          std::uint64_t &retained);

} // namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail
