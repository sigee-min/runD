#pragma once

#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract::kernel_case {

struct ResidentRunFixture {
  rund::AccelContext context{};
  rund::AccelBuffer input{};
  rund::AccelBuffer output{};
  rund::AccelKernel kernel{};
  std::array<rund::kernel::i32, 8u> host_input{};
  std::uint64_t expected_hash = 0u;
  bool ok = false;
};

[[nodiscard]] std::array<rund::AccelRunBinding, 2u>
Bindings(const ResidentRunFixture &fixture) noexcept;

[[nodiscard]] rund::AccelRun
RunRequest(const std::array<rund::AccelRunBinding, 2u> &bindings,
           std::size_t tile_count, bool fresh_evidence) noexcept;

[[nodiscard]] ResidentRunFixture
MakeResidentRunFixture(const rund::AccelContext &context);

[[nodiscard]] bool
ResidentRunEvidenceMatches(const ResidentRunFixture &fixture);

[[nodiscard]] bool
WindowedResidentRunMatches(const rund::AccelContext &context);

[[nodiscard]] bool
MultiWriteResidentRunMatches(const rund::AccelContext &context);

[[nodiscard]] bool
ResidentRunRejectsForgedAndForeign(const ResidentRunFixture &fixture);

[[nodiscard]] bool
LogicalAliasAdmissionMatches(const ResidentRunFixture &fixture);

[[nodiscard]] bool
IndexedWriteBoundaryMatches(const rund::AccelContext &context);

} // namespace node_accel_contract::kernel_case
