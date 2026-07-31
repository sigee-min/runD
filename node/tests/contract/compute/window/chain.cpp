#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace rund::node::test_contract::window {

[[nodiscard]] int CheckWindowChain(Device &device, const Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, kMatrixWidth> first_seed{10u, 20u, 30u,
                                                               40u};
  constexpr std::array<std::uint32_t, kMatrixWidth> second_seed{100u, 200u,
                                                                300u, 400u};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> count_values{
      static_cast<std::uint32_t>(kMatrixMaximum)};
  constexpr std::array<std::uint32_t, 1u> safe_failure{
      static_cast<std::uint32_t>(kMatrixWindows)};
  constexpr std::array<std::uint32_t, kMatrixBacking> first_initial{
      901u, 902u, 903u, 904u, 905u, 906u, 907u, 908u, 909u};
  constexpr std::array<std::uint32_t, kMatrixBacking> second_initial{
      801u, 802u, 803u, 804u, 805u, 806u, 807u, 808u, 809u};

  auto body = MatrixBody(device);
  auto consumer = on(device)
                      .input<std::uint32_t>(kMatrixWidth)
                      .zip_input<std::uint32_t>(kMatrixWidth)
                      .branch([](auto first, auto second) {
                        return first.combine(
                            "resident-window-chain-consume", second,
                            [](auto left, auto right) { return left + right; });
                      })
                      .compile();
  auto first =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{first_seed});
  auto second =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{second_seed});
  auto witness = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{witness_values});
  auto fail_at = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{safe_failure});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto first_backing = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{first_initial});
  auto second_backing = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{second_initial});
  auto derived = device.buffer<std::uint32_t>(kMatrixWidth);
  if (!body || !consumer || !first || !second || !witness || !fail_at ||
      !count || !first_backing || !second_backing || !derived) {
    return 1;
  }
  auto first_view = first_backing->view(1u, kMatrixWidth, 2u);
  auto second_view = second_backing->view(0u, kMatrixWidth, 2u);
  if (!first_view || !second_view) {
    return 2;
  }

  auto builder = pipeline(device);
  builder
      .windows<kMatrixMaximum, kMatrixTile>(
          *body, rund::compute::window(*count),
          read(*first, *second, *witness, *fail_at),
          write(*first_view, *second_view))
      .then(*consumer, read(*first_view, *second_view), write(*derived));
  const auto plan = builder.plan();
  if (!plan || plan->publish_count != 2u ||
      plan->publish_bytes != 2u * kMatrixWidth * sizeof(std::uint32_t)) {
    std::fprintf(
        stderr,
        "window chain plan backend=%u status=%u reason=%u publishes=%llu "
        "bytes=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(plan.ok()),
        static_cast<unsigned>(plan.reason()),
        static_cast<unsigned long long>(plan ? plan->publish_count : 0u),
        static_cast<unsigned long long>(plan ? plan->publish_bytes : 0u));
    return 3;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared) {
    std::fprintf(stderr, "window chain prepare backend=%u reason=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(prepared.reason()));
    return 4;
  }
  const std::shared_ptr<rund::compute::detail::PipelineState> state =
      rund::compute::detail::PipelineStateAccess::state(*prepared);
  constexpr std::array<std::uint16_t, kMatrixWindows + 2u> expected_rank{
      0u, 1u, 2u, 3u, 3u};
  if (state == nullptr || state->window_rank.size() != expected_rank.size() ||
      !std::equal(state->window_rank.begin(), state->window_rank.end(),
                  expected_rank.begin())) {
    return 5;
  }
  const Status ran = prepared->run();
  const Stats stats = prepared->stats();
  std::array<std::uint32_t, kMatrixBacking> first_actual{};
  std::array<std::uint32_t, kMatrixBacking> second_actual{};
  std::array<std::uint32_t, kMatrixWidth> derived_actual{};
  if (!ran || !prepared->read(*first_backing, first_actual) ||
      !prepared->read(*second_backing, second_actual) ||
      !prepared->read(*derived, derived_actual)) {
    std::fprintf(stderr,
                 "window chain run backend=%u status=%u reason=%u "
                 "poison=%u generation=%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(ran.ok()),
                 static_cast<unsigned>(ran.reason()),
                 static_cast<unsigned>(prepared->poisoned()),
                 static_cast<unsigned long long>(prepared->generation()));
    return 5;
  }
  constexpr std::uint32_t sum =
      static_cast<std::uint32_t>(kMatrixMaximum * (kMatrixMaximum - 1u) / 2u);
  std::array<std::uint32_t, kMatrixWidth> expected_first{};
  std::array<std::uint32_t, kMatrixWidth> expected_second{};
  std::array<std::uint32_t, kMatrixWidth> expected_derived{};
  for (std::size_t index = 0u; index < kMatrixWidth; ++index) {
    expected_first[index] = first_seed[index] + sum;
    expected_second[index] = second_seed[index] + sum + sum;
    expected_derived[index] = expected_first[index] + expected_second[index];
  }
  const auto expected_first_backing =
      Projected(first_initial, expected_first, 1u, 2u);
  const auto expected_second_backing =
      Projected(second_initial, expected_second, 0u, 2u);
  if (first_actual != expected_first_backing ||
      second_actual != expected_second_backing ||
      derived_actual != expected_derived || stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "window chain payload backend=%u first=%u/%u second=%u/%u "
        "derived=%u/%u steps=%llu verified=%llu submits=%llu\n",
        static_cast<unsigned>(backend), first_actual[1u], expected_first[0],
        second_actual[0u], expected_second[0], derived_actual[0u],
        expected_derived[0],
        static_cast<unsigned long long>(stats.pipeline.step_count),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 6;
  }
  return 0;
}

} // namespace rund::node::test_contract::window
