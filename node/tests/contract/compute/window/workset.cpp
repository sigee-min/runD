#include "../pipeline/local.hpp"
#include "local.hpp"
#include "workset/local.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace rund::node::test_contract::window {
namespace {

constexpr std::size_t kMaximum = 10u;
constexpr std::size_t kTile = 4u;
constexpr std::size_t kInner = 3u;
constexpr std::size_t kDomain = 64u;
constexpr std::size_t kOuter = CeilDiv(kMaximum, kTile);
constexpr std::uint32_t kOuterSeed = 7u;
constexpr std::array<std::uint32_t, kDomain> kDomainValues = [] {
  std::array<std::uint32_t, kDomain> values{};
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = static_cast<std::uint32_t>(3u * index + 1u);
  }
  return values;
}();
constexpr std::array<std::uint32_t, 0u> kEmpty{};
constexpr std::array<std::uint32_t, 3u> kSparse{1u, 47u, 63u};
constexpr std::array<std::uint32_t, 5u> kTail{0u, 1u, 31u, 47u, 63u};
constexpr std::array<std::uint32_t, kMaximum> kFull{0u,  1u,  2u,  5u,  31u,
                                                    47u, 48u, 60u, 62u, 63u};
constexpr std::array<std::uint32_t, kMaximum + 1u> kOverflow{
    0u, 1u, 2u, 3u, 5u, 31u, 47u, 48u, 60u, 62u, 63u};

template <std::size_t N>
[[nodiscard]] consteval bool
Canonical(const std::array<std::uint32_t, N> &ordinals) noexcept {
  for (std::size_t index = 0u; index < ordinals.size(); ++index) {
    if (ordinals[index] >= kDomain ||
        (index != 0u && ordinals[index - 1u] >= ordinals[index])) {
      return false;
    }
  }
  return true;
}

static_assert(Canonical(kEmpty));
static_assert(Canonical(kSparse));
static_assert(Canonical(kTail));
static_assert(Canonical(kFull));
static_assert(Canonical(kOverflow));

[[nodiscard]] auto WorksetProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .map<std::uint32_t>("resident-workset-flags", kDomain,
                          [](auto flag) { return flag; })
      .compact({.capacity = kMaximum})
      .compile();
}

[[nodiscard]] auto SeedProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kMaximum)
      .zip_input<std::uint32_t>(kDomain)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto queue, auto domain, auto total, auto ordinal) {
        auto current = resident<kMaximum, kTile>(total, ordinal);
        auto active_ordinals = queue.gather(current.items());
        auto sum = domain.gather(active_ordinals).reduce(Reduce::Sum);
        return outputs(sum, current.count());
      })
      .compile();
}

[[nodiscard]] auto ActionProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto value, auto tile_count) {
        return value.combine(
            "resident-workset-action", tile_count,
            [](auto current, auto count) { return current + count; });
      })
      .compile();
}

[[nodiscard]] auto FoldProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto tile_count) {
        auto checked = tile.combine(
            "resident-workset-tail", tile_count,
            [](auto value, auto count) { return value + count * 0u; });
        return outer.combine(
            "resident-workset-fold", checked,
            [](auto left, auto right) { return left + right; });
      })
      .compile();
}

[[nodiscard]] auto PublishProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .map<std::uint32_t>("resident-workset-publish", 1u,
                          [](auto value) { return value; })
      .compile();
}

template <std::size_t N>
[[nodiscard]] constexpr std::array<std::uint32_t, kDomain>
Flags(const std::array<std::uint32_t, N> &ordinals) noexcept {
  std::array<std::uint32_t, kDomain> flags{};
  for (const std::uint32_t ordinal : ordinals) {
    flags[ordinal] = 1u;
  }
  return flags;
}

[[nodiscard]] constexpr std::uint32_t
Delta(const std::span<const std::uint32_t> ordinals) noexcept {
  std::uint32_t result{};
  for (const std::uint32_t ordinal : ordinals) {
    result += kDomainValues[ordinal];
  }
  result += static_cast<std::uint32_t>(ordinals.size() * kInner);
  return result;
}

[[nodiscard]] bool WarmSetupClean(const rund::compute::Stats &stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u && stats.uploaded_bytes == 0u &&
         stats.download_events == 0u && stats.downloaded_bytes == 0u &&
         stats.pipeline.rebinding_count == 0u && stats.output_hash == 0u;
}

template <std::size_t N>
[[nodiscard]] int
CheckPublished(rund::compute::Pipeline &pipeline,
               rund::compute::Buffer<std::uint32_t> &flags,
               const rund::compute::Buffer<std::uint32_t> &queue,
               const rund::compute::Buffer<std::uint32_t> &count,
               const rund::compute::Buffer<std::uint32_t> &output,
               const rund::compute::Backend backend,
               const std::array<std::uint32_t, N> &ordinals,
               const std::uint32_t expected_output,
               const std::uint64_t generation, const std::uint64_t discards) {
  using namespace rund::compute;
  const auto flag_values = Flags(ordinals);
  if (!rund_node_test_pipeline::Overwrite(flags, flag_values)) {
    return 1;
  }

  const MemoryStats before = pipeline.memory();
  const Status ran = pipeline.run();
  // Capture execution evidence before the explicit test-only observations
  // below. A zero transfer/readback row therefore proves that the resident
  // count crossed the Program/window boundary without host intervention.
  const Stats stats = pipeline.stats();
  const MemoryStats after = pipeline.memory();
  std::array<std::uint32_t, kMaximum> actual_queue{};
  std::array<std::uint32_t, 1u> actual_count{};
  std::array<std::uint32_t, 1u> actual_output{};
  const bool observed = pipeline.read(queue, actual_queue) &&
                        pipeline.read(count, actual_count) &&
                        pipeline.read(output, actual_output);
  const std::span<const std::uint32_t> expected{ordinals};
  const std::uint64_t active_windows = CeilDiv(expected.size(), kTile);
  const std::uint64_t expected_submits = backend == Backend::Cpu ? 0u : 1u;
  const bool queue_matches =
      std::equal(expected.begin(), expected.end(), actual_queue.begin());
  const bool valid =
      ran && observed && queue_matches && actual_count[0] == expected.size() &&
      actual_output[0] == expected_output &&
      pipeline.generation() == generation && !pipeline.poisoned() &&
      stats.command_submits == expected_submits &&
      stats.pipeline.step_count == 3u &&
      stats.pipeline.verified_step_count == 3u &&
      stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
      stats.pipeline.executed_outer_window_count == active_windows &&
      stats.pipeline.skipped_outer_window_count == kOuter - active_windows &&
      stats.pipeline.executed_inner_iteration_count ==
          active_windows * kInner &&
      stats.pipeline.skipped_inner_iteration_count ==
          (kOuter - active_windows) * kInner &&
      stats.publication.generation == generation &&
      stats.publication.commit_count == generation &&
      stats.publication.discard_count == discards && WarmSetupClean(stats) &&
      rund_node_test_pipeline::SameMemory(before, after);
  if (!valid) {
    std::fprintf(
        stderr,
        "resident workset run backend=%u count=%zu status=%u/%u "
        "observed=%u queue=%u count=%u output=%u/%u generation=%llu/%llu "
        "poison=%u submits=%llu steps=%llu/%llu failed=%llu "
        "outer=%llu/%llu inner=%llu/%llu generated=%llu/%llu "
        "publication=%llu/%llu/%llu warm=%u memory=%u\n",
        static_cast<unsigned>(backend), expected.size(),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        static_cast<unsigned>(observed), static_cast<unsigned>(queue_matches),
        actual_count[0], actual_output[0], expected_output,
        static_cast<unsigned long long>(pipeline.generation()),
        static_cast<unsigned long long>(generation),
        static_cast<unsigned>(pipeline.poisoned()),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.pipeline.step_count),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(stats.control.generated_item_count),
        static_cast<unsigned long long>(stats.control.generated_capacity),
        static_cast<unsigned long long>(stats.publication.generation),
        static_cast<unsigned long long>(stats.publication.commit_count),
        static_cast<unsigned long long>(stats.publication.discard_count),
        static_cast<unsigned>(WarmSetupClean(stats)),
        static_cast<unsigned>(
            rund_node_test_pipeline::SameMemory(before, after)));
  }
  return valid ? 0 : 2;
}

[[nodiscard]] int
CheckOverflow(rund::compute::Pipeline &pipeline,
              rund::compute::Buffer<std::uint32_t> &flags,
              const rund::compute::Buffer<std::uint32_t> &queue,
              const rund::compute::Buffer<std::uint32_t> &count,
              const rund::compute::Buffer<std::uint32_t> &output,
              const rund::compute::Backend backend,
              const std::uint32_t expected_output) {
  using namespace rund::compute;
  const auto overflow_flags = Flags(kOverflow);
  if (!rund_node_test_pipeline::Overwrite(flags, overflow_flags)) {
    return 1;
  }

  const MemoryStats before = pipeline.memory();
  const Status failed = pipeline.run();
  // Keep failure evidence independent of the explicit retained-state read.
  const Stats stats = pipeline.stats();
  const MemoryStats after = pipeline.memory();
  std::array<std::uint32_t, kMaximum> rejected_queue{};
  std::array<std::uint32_t, 1u> rejected_count{};
  std::array<std::uint32_t, 1u> retained_output{};
  const Status queue_read = pipeline.read(queue, rejected_queue);
  const Status count_read = pipeline.read(count, rejected_count);
  const bool retained =
      static_cast<bool>(pipeline.read(output, retained_output));
  const std::uint64_t expected_submits = backend == Backend::Cpu ? 0u : 1u;
  const bool valid =
      !failed && failed.reason() == Reason::CompactCapacityInsufficient &&
      pipeline.poisoned() && pipeline.generation() == 5u && retained &&
      !queue_read && !count_read && retained_output[0] == expected_output &&
      stats.command_submits == expected_submits &&
      stats.pipeline.step_count == 3u &&
      stats.pipeline.verified_step_count == 0u &&
      stats.pipeline.failed_step_index == 0u &&
      stats.pipeline.executed_outer_window_count == 0u &&
      stats.pipeline.executed_inner_iteration_count == 0u &&
      stats.publication.generation == 5u &&
      stats.publication.commit_count == 5u &&
      stats.publication.discard_count == 1u && WarmSetupClean(stats) &&
      rund_node_test_pipeline::SameMemory(before, after);
  if (!valid) {
    std::fprintf(
        stderr,
        "resident workset overflow backend=%u status=%u/%u poison=%u "
        "generation=%llu retained=%u queue_read=%u/%u count_read=%u/%u "
        "output=%u/%u "
        "submits=%llu steps=%llu/%llu failed=%llu outer=%llu inner=%llu "
        "generated=%llu/%llu publication=%llu/%llu/%llu warm=%u memory=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(failed.ok()),
        static_cast<unsigned>(failed.reason()),
        static_cast<unsigned>(pipeline.poisoned()),
        static_cast<unsigned long long>(pipeline.generation()),
        static_cast<unsigned>(retained), static_cast<unsigned>(queue_read.ok()),
        static_cast<unsigned>(queue_read.reason()),
        static_cast<unsigned>(count_read.ok()),
        static_cast<unsigned>(count_read.reason()), retained_output[0],
        expected_output, static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.pipeline.step_count),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(stats.control.generated_item_count),
        static_cast<unsigned long long>(stats.control.generated_capacity),
        static_cast<unsigned long long>(stats.publication.generation),
        static_cast<unsigned long long>(stats.publication.commit_count),
        static_cast<unsigned long long>(stats.publication.discard_count),
        static_cast<unsigned>(WarmSetupClean(stats)),
        static_cast<unsigned>(
            rund_node_test_pipeline::SameMemory(before, after)));
  }
  return valid ? 0 : 2;
}

} // namespace

[[nodiscard]] int CheckResidentWorkset(rund::compute::Device &device,
                                       const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial_output{kOuterSeed};
  constexpr auto empty_flags = Flags(kEmpty);
  auto workset = WorksetProgram(device);
  auto seed = SeedProgram(device);
  auto action = ActionProgram(device);
  auto fold = FoldProgram(device);
  auto publish = PublishProgram(device);
  auto flags = device.upload<std::uint32_t>(empty_flags);
  auto domain = device.upload<std::uint32_t>(kDomainValues);
  auto queue = device.buffer<std::uint32_t>(kMaximum);
  auto count = device.buffer<std::uint32_t>(1u);
  auto nested_output = device.buffer<std::uint32_t>(1u);
  auto output_first = device.upload<std::uint32_t>(initial_output);
  auto output_second = device.upload<std::uint32_t>(initial_output);
  if (!workset || !seed || !action || !fold || !publish || !flags || !domain ||
      !queue || !count || !nested_output || !output_first || !output_second) {
    return 1;
  }

  const auto body = tile_repeat<kInner>(*seed, *action, *fold);
  auto builder = pipeline(device);
  builder.state(*output_first, *output_second)
      .then(*workset, read(*flags), write(*queue, *count))
      .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                read(*output_first, *queue, *domain),
                                write(*nested_output))
      .then(*publish, read(*nested_output), write(*output_second))
      .commit();
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != kOuter ||
      plan->tile_capacity != kTile || plan->inner_iteration_count != kInner) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared || prepared->plan() != *plan) {
    const Location location = prepared.location();
    std::fprintf(stderr,
                 "resident workset prepare backend=%u status=%u reason=%u "
                 "location=%u/%u/%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(prepared.ok()),
                 static_cast<unsigned>(prepared.reason()), location.step,
                 location.iteration, location.node);
    return 3;
  }

  std::uint32_t expected_output = kOuterSeed;
  expected_output += Delta(kEmpty);
  if (const int result =
          CheckPublished(*prepared, *flags, *queue, *count, *output_second,
                         backend, kEmpty, expected_output, 1u, 0u);
      result != 0) {
    return 10 + result;
  }
  expected_output += Delta(kSparse);
  if (const int result =
          CheckPublished(*prepared, *flags, *queue, *count, *output_second,
                         backend, kSparse, expected_output, 2u, 0u);
      result != 0) {
    return 20 + result;
  }
  expected_output += Delta(kTail);
  if (const int result =
          CheckPublished(*prepared, *flags, *queue, *count, *output_second,
                         backend, kTail, expected_output, 3u, 0u);
      result != 0) {
    return 30 + result;
  }
  expected_output += Delta(kFull);
  if (const int result =
          CheckPublished(*prepared, *flags, *queue, *count, *output_second,
                         backend, kFull, expected_output, 4u, 0u);
      result != 0) {
    return 40 + result;
  }
  expected_output += Delta(kFull);
  if (const int result =
          CheckPublished(*prepared, *flags, *queue, *count, *output_second,
                         backend, kFull, expected_output, 5u, 0u);
      result != 0) {
    return 50 + result;
  }
  if (const int result =
          CheckOverflow(*prepared, *flags, *queue, *count, *output_second,
                        backend, expected_output);
      result != 0) {
    return 60 + result;
  }
  return 0;
}

} // namespace rund::node::test_contract::window
