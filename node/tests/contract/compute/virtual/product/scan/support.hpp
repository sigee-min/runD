#pragma once

#include "../backing.hpp"
#include "../model.hpp"
#include "../route.hpp"

#include "../../../../target/selection.hpp"

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product::scan {

inline constexpr std::size_t ScanFrameElements = 16u;
inline constexpr std::size_t ScanElements = 53u;
inline constexpr std::size_t ExclusivePayload = ScanFrameElements - 1u;

using ScanGenerations = std::array<std::uint64_t, 2u>;

struct ScanControlIdentity final {
  std::uint64_t generation{};
  std::uint8_t parity{};
  bool poisoned{};

  [[nodiscard]] friend bool operator==(const ScanControlIdentity &,
                                       const ScanControlIdentity &) = default;
};

using ScanControls = std::array<ScanControlIdentity, 2u>;

class PersistentScanBacking final : public rund::compute::VirtualBacking {
public:
  explicit PersistentScanBacking(std::size_t bytes);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] rund::compute::VirtualBackingTier
  tier() const noexcept override;
  [[nodiscard]] std::uint32_t max_parallel_reads() const noexcept override;
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;
  [[nodiscard]] bool seed(std::span<const std::byte> input) noexcept;
  [[nodiscard]] std::uint64_t read_bytes() const noexcept;

private:
  std::vector<std::byte> bytes_;
  std::atomic<std::uint64_t> read_bytes_{};
};

using U32Program = rund::compute::Program<std::uint32_t(std::uint32_t)>;
using U32Pipeline =
    rund::compute::VirtualPipeline<std::uint32_t(std::uint32_t)>;

struct ScanFixture final {
  rund::compute::Backend backend{rund::compute::Backend::Cpu};
  std::optional<rund::compute::Device> device{};
  std::optional<U32Program> inclusive_program{};
  std::optional<U32Program> exclusive_program{};
  std::shared_ptr<MemoryVirtualBacking> input_backing{};
  std::shared_ptr<MemoryVirtualBacking> inclusive_backing{};
  std::optional<U32Pipeline> inclusive{};
  std::array<std::uint32_t, ScanElements> values{};
  std::unique_ptr<DeviceVsmBypassScope> tiered_route{};
};

[[nodiscard]] bool uses_device_vsm(
    rund::compute::Backend backend,
    const rund::compute::ResidencyStats &stats) noexcept;
[[nodiscard]] ScanGenerations scan_generations(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState>
        &state) noexcept;
[[nodiscard]] ScanControls scan_controls(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState>
        &state) noexcept;
[[nodiscard]] bool scan_matches(MemoryVirtualBacking &backing,
                                std::span<const std::uint32_t> input,
                                bool inclusive) noexcept;

[[nodiscard]] int InitializeScanFixture(ScanFixture &fixture,
                                         rund::compute::Backend backend);
[[nodiscard]] int BeginTieredRoute(ScanFixture &fixture);
void ResetTieredRoute(ScanFixture &fixture) noexcept;

[[nodiscard]] int RunBasic(ScanFixture &fixture);
[[nodiscard]] int RunTieredCold(ScanFixture &fixture);
[[nodiscard]] int RunOverflow(ScanFixture &fixture);
[[nodiscard]] int RunTieredWide(ScanFixture &fixture);
[[nodiscard]] int RunUnknown(ScanFixture &fixture);
[[nodiscard]] int RunPoisonRecovery(ScanFixture &fixture);

} // namespace rund_node_test_virtual::product::scan
