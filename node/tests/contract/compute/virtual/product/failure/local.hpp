#pragma once

#include "../backing.hpp"
#include "../model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product::failure {

using I32Program = rund::compute::Program<std::int32_t(std::int32_t)>;
using I32Pipeline =
    rund::compute::VirtualPipeline<std::int32_t(std::int32_t)>;
using I32Buffer = rund::compute::VirtualBuffer<std::int32_t>;

struct FailureFixture final {
  rund::compute::Backend backend{rund::compute::Backend::Cpu};
  std::optional<rund::compute::Device> device{};
  std::optional<I32Program> program{};
  std::shared_ptr<MemoryVirtualBacking> input_backing{};
  std::shared_ptr<MemoryVirtualBacking> output_backing{};
  std::optional<I32Buffer> input{};
  std::optional<I32Buffer> output{};
  std::optional<I32Pipeline> prepared{};
  std::array<std::int32_t, LogicalElements> seeded{};
};

class OffsetFailPersistentBacking final : public rund::compute::VirtualBacking {
public:
  explicit OffsetFailPersistentBacking(
      std::span<const std::int32_t> values);
  OffsetFailPersistentBacking(std::span<const std::byte> bytes,
                              std::uint64_t failure_offset);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] rund::compute::VirtualBackingTier
  tier() const noexcept override;
  [[nodiscard]] std::uint32_t max_parallel_reads() const noexcept override;
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;

private:
  std::vector<std::byte> bytes_;
  std::uint64_t failure_offset_{};
  std::atomic<bool> fail_{true};
};

[[nodiscard]] int InitializeFailureFixture(FailureFixture &fixture,
                                            rund::compute::Backend backend);
[[nodiscard]] int CheckReturnedFailureBoundary();
[[nodiscard]] int CheckGenericBackingFailure(FailureFixture &fixture);
[[nodiscard]] int CheckDeviceVsmFailure(FailureFixture &fixture);
[[nodiscard]] int CheckNativeTransferFailure(FailureFixture &fixture);
[[nodiscard]] int CheckMultiLaneFailure(FailureFixture &fixture);

} // namespace rund_node_test_virtual::product::failure
