#pragma once

#include "../local.hpp"

#include "../../../target/selection.hpp"
#include "../../allocation.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/local.hpp"
#include "src/compute/pipeline/state.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace rund_node_test_pipeline::checkpoint {

inline constexpr std::array<std::int32_t, 4u> Initial{1, 2, 3, 4};
inline constexpr std::array<std::int32_t, 4u> Thrice{4, 5, 6, 7};
inline constexpr std::array<std::int32_t, 4u> Fourth{5, 6, 7, 8};

struct Context final {
  rund::compute::Device &device;
  Backend backend;
  rund::compute::Program<std::int32_t(std::int32_t)> advance;
  Buffer<std::int32_t> first;
  Buffer<std::int32_t> second;
  Pipeline prepared;
  std::shared_ptr<rund::compute::detail::PipelineState> source_state;
  rund::compute::SnapshotStorage storage;
  rund::compute::SnapshotStorage small;
  rund::compute::LatestDeviceState latest;
  std::optional<Pipeline> parity_zero{};
  std::optional<Pipeline> parity_one{};
  std::shared_ptr<rund::compute::detail::PipelineState> parity_zero_state{};
  std::array<std::int32_t, Initial.size()> observed{};
};

struct Preparation final {
  std::optional<Context> context{};
  int error{};
};

[[nodiscard]] Preparation Prepare(rund::compute::Device &, Backend);
[[nodiscard]] int CheckInitialAndParity(Context &);
[[nodiscard]] int CheckBusyAndCopy(Context &);
[[nodiscard]] int CheckAliasRejection(Context &);
[[nodiscard]] int CheckStorageCapacity(Context &);
[[nodiscard]] int CheckPortability(Context &);
[[nodiscard]] int CheckBoundaries(Context &);

[[nodiscard]] bool
SameCheckpointStats(const rund::compute::CheckpointStats &,
                    const rund::compute::CheckpointStats &) noexcept;

} // namespace rund_node_test_pipeline::checkpoint
