#pragma once

#include <rund/reason.hpp>

#include <cstdint>
#include <string_view>

namespace rund {

struct PreparedMemory final {
  struct Capacity final {
    ReasonCode code = ReasonCode::PreparedCapacityNotProved;
    std::uint32_t worker_count = 0u;
    std::uint64_t requested_lane_capacity = 0u;
    std::uint64_t available_lane_capacity = 0u;
    std::uint64_t requested_reduction_input_capacity = 0u;
    std::uint64_t available_reduction_input_capacity = 0u;
    std::uint64_t requested_reduction_output_capacity = 0u;
    std::uint64_t available_reduction_output_capacity = 0u;
    std::uint64_t requested_epoch_workers = 0u;
    std::uint64_t available_epoch_workers = 0u;

    [[nodiscard]] constexpr bool ok() const noexcept {
      return code == ReasonCode::Ok;
    }
    [[nodiscard]] bool valid() const noexcept {
      return ValidPreparedMemoryReason(code);
    }
    constexpr explicit operator bool() const noexcept { return ok(); }
    [[nodiscard]] std::string_view error() const noexcept {
      return ok() ? std::string_view{} : ReasonString(code);
    }
    [[nodiscard]] constexpr int exit_code() const noexcept {
      return ok() ? 0 : 1;
    }
  };

  Capacity capacity{};
  std::uint64_t lane_capacity = 0u;
  std::uint64_t lane_high_water = 0u;
  std::uint64_t lane_overflow_count = 0u;
  std::uint64_t reduction_input_capacity = 0u;
  std::uint64_t reduction_output_capacity = 0u;
  std::uint64_t reduction_input_high_water = 0u;
  std::uint64_t reduction_output_high_water = 0u;
  std::uint64_t reduction_overflow_count = 0u;
  std::uint64_t epoch_current = 0u;
  std::uint64_t epoch_reclaimable = 0u;
};

bool record_memory(const PreparedMemory &memory) noexcept;

} // namespace rund
