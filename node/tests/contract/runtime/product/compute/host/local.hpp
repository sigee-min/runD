#pragma once

#include <rund/compute/session.hpp>

#include <cstdint>

namespace runtime_compute_host_detail {

struct ReadyOrder final {
  std::uint64_t allocations{};
  bool ok{};
};

[[nodiscard]] ReadyOrder CheckReadyOrder(std::uint32_t workers);
[[nodiscard]] bool CheckTimedSubmissionObservation();
[[nodiscard]] bool CheckCpuStepParity(rund::Session &runtime);
[[nodiscard]] bool CheckServerReplay(rund::Session &server);

} // namespace runtime_compute_host_detail
