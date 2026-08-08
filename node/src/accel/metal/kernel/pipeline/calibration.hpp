#pragma once

#include "icb.hpp"

#include <cstdint>
#include <mutex>
#include <type_traits>

namespace rund::node::accel::detail {

// PickMetal's system-default selection retains one fixed last-device entry
// keyed by exact registry identity. The probe stays injected so every cache
// transition is independently testable without a Metal Device.
class MetalIcbCalibrationCache final {
public:
  MetalIcbCalibrationCache() noexcept = default;

  template <class Probe>
  [[nodiscard]] MetalIcbCalibration load(const std::uint64_t registry_id,
                                         Probe &&probe) noexcept {
    static_assert(
        std::is_nothrow_invocable_r_v<bool, Probe &, MetalIcbCalibration &>);
    MetalIcbCalibration calibration{};
    if (registry_id == 0u) {
      (void)probe_valid(probe, calibration);
      return calibration;
    }

    try {
      std::lock_guard<std::mutex> lock{mutex_};
      if (occupied_ && registry_id_ == registry_id) {
        return calibration_;
      }
      if (probe_valid(probe, calibration)) {
        calibration_ = calibration;
        registry_id_ = registry_id;
        occupied_ = true;
      }
      return calibration;
    } catch (...) {
      // A lock failure cannot add a throw path to adapter opening. The exact
      // Device remains usable through an uncached probe.
      (void)probe_valid(probe, calibration);
      return calibration;
    }
  }

private:
  template <class Probe>
  [[nodiscard]] static bool
  probe_valid(Probe &probe, MetalIcbCalibration &calibration) noexcept {
    calibration = {};
    if (!probe(calibration) || !ValidMetalIcbCalibration(calibration)) {
      calibration = {};
      return false;
    }
    return true;
  }

  std::mutex mutex_{};
  MetalIcbCalibration calibration_{};
  std::uint64_t registry_id_{};
  bool occupied_{};
};

} // namespace rund::node::accel::detail
