#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

namespace {

struct Value final {};
struct Doubled final {};

using rund::compute::Buffer;
using rund::compute::LatestDeviceState;
using rund::compute::Pipeline;
using rund::compute::PipelineBuilder;
using rund::compute::SnapshotStorage;
using rund::compute::StateSnapshot;

static_assert(!std::copy_constructible<Pipeline>);
static_assert(std::is_nothrow_move_constructible_v<Pipeline>);
static_assert(!std::copy_constructible<PipelineBuilder>);
static_assert(std::is_nothrow_move_constructible_v<PipelineBuilder>);
static_assert(std::copy_constructible<StateSnapshot>);
static_assert(std::copy_constructible<LatestDeviceState>);
static_assert(!std::copy_constructible<SnapshotStorage>);
static_assert(std::is_nothrow_move_constructible_v<SnapshotStorage>);
static_assert(std::same_as<decltype(std::declval<rund::Session &>().compute(
                               std::declval<Pipeline &>())),
                           rund::compute::Request>);

template <class T>
concept ReadsTemporary =
    requires(T value) { rund::compute::read(std::move(value)); };

template <class T>
concept WritesConst = requires(const T value) { rund::compute::write(value); };

template <class T>
concept ViewsTemporary = requires(T value) { std::move(value).view(); };

template <class T>
concept WritesConstView =
    requires(const T &value) { rund::compute::write(value.view()); };

template <class T>
concept WritesMutableView =
    requires(T &value) { rund::compute::write(value.view()); };

static_assert(!ReadsTemporary<Buffer<std::int32_t>>);
static_assert(!WritesConst<Buffer<std::int32_t>>);
static_assert(!ViewsTemporary<Buffer<std::int32_t>>);
static_assert(!WritesConstView<Buffer<std::int32_t>>);
static_assert(WritesMutableView<Buffer<std::int32_t>>);

[[nodiscard]] int DependentWide(rund::compute::Device &device) {
  using Real = rund::compute::Fixed<20, 44>;
  constexpr std::array<Real, 4u> position{Real::from_raw(3), Real::from_raw(6),
                                          Real::from_raw(9),
                                          Real::from_raw(12)};
  constexpr std::array<Real, 4u> velocity{Real::from_raw(2), Real::from_raw(4),
                                          Real::from_raw(6), Real::from_raw(8)};
  constexpr std::array<Real, 4u> force{Real::from_raw(1), Real::from_raw(3),
                                       Real::from_raw(5), Real::from_raw(7)};

  auto integrate = rund::compute::on(device)
                       .input<Real>(position.size())
                       .zip_input<Real>(velocity.size())
                       .zip_input<Real>(force.size())
                       .map("pipeline-installed-integrate",
                            [](auto p, auto v, auto f) {
                              return rund::compute::quantize<Real>(p + v + f);
                            })
                       .compile();
  auto advance =
      rund::compute::on(device)
          .map<Real>("pipeline-installed-advance", position.size(),
                     [](auto value) {
                       return rund::compute::quantize<Real>(value + value);
                     })
          .compile();
  auto p = device.upload<Real>(position);
  auto v = device.upload<Real>(velocity);
  auto f = device.upload<Real>(force);
  auto middle = device.buffer<Real>(position.size());
  auto output = device.buffer<Real>(position.size());
  if (!integrate || !advance || !p || !v || !f || !middle || !output) {
    return 1;
  }

  auto prepared = rund::compute::pipeline(device)
                      .profile(rund::compute::PipelineProfile::Steps)
                      .then(*integrate, rund::compute::read(*p, *v, *f),
                            rund::compute::write(*middle))
                      .then(*advance, rund::compute::read(*middle),
                            rund::compute::write(*output))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  Pipeline pipeline = std::move(prepared).value();
  const auto ran = pipeline.run();
  if (!ran) {
    return ran.exit_code();
  }
  std::array<rund::compute::PipelineStepProfile, 2u> steps{};
  const auto profile = pipeline.profile(steps);
  if (!profile) {
    return profile.exit_code();
  }
  std::array<Real, position.size()> observed{};
  const auto read = pipeline.read(*output, observed);
  if (!read) {
    return read.exit_code();
  }
  return observed == std::array<Real, 4u>{Real::from_raw(12),
                                          Real::from_raw(26),
                                          Real::from_raw(40),
                                          Real::from_raw(54)} &&
                 pipeline.stats().pipeline.step_count == 2u &&
                 pipeline.stats().pipeline.resource_count == 5u &&
                 pipeline.stats().pipeline.barrier_count == 1u &&
                 pipeline.stats().command_submits == 0u &&
                 profile->written == steps.size() &&
                 profile->total == steps.size() && !profile->truncated() &&
                 steps[0].index == 0u && steps[1].index == 1u &&
                 steps[0].program == integrate->fingerprint() &&
                 steps[1].program == advance->fingerprint() &&
                 steps[0].execution.available() &&
                 steps[1].execution.available() &&
                 steps[0].timing.available() && steps[1].timing.available() &&
                 steps[0].timing.clock ==
                     rund::compute::StepClock::HostSteady &&
                 steps[1].timing.clock ==
                     rund::compute::StepClock::HostSteady &&
                 profile->memory.available() &&
                 profile->shared_memory.available() &&
                 profile->referenced_resource_bytes ==
                     sizeof(Real) * position.size() * 5u &&
                 profile->instrumentation_command_count == 0u &&
                 profile->instrumentation_byte_count == 0u &&
                 profile->observation.available()
             ? 0
             : 2;
}

[[nodiscard]] int MultiInput(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> first{1, 2, 3, 4};
  constexpr std::array<std::int32_t, 4u> second{10, 20, 30, 40};
  constexpr std::array<std::int32_t, 4u> third{100, 200, 300, 400};

  auto program = rund::compute::on(device)
                     .input<std::int32_t>(first.size())
                     .zip_input<std::int32_t>(second.size())
                     .zip_input<std::int32_t>(third.size())
                     .map("pipeline-multi-input-sum",
                          [](auto a, auto b, auto c) { return a + b + c; })
                     .compile();
  auto a = device.upload<std::int32_t>(first);
  auto b = device.upload<std::int32_t>(second);
  auto c = device.upload<std::int32_t>(third);
  auto sum = device.buffer<std::int32_t>(first.size());
  if (!program) {
    return program.exit_code();
  }
  if (!a) {
    return a.exit_code();
  }
  if (!b) {
    return b.exit_code();
  }
  if (!c) {
    return c.exit_code();
  }
  if (!sum) {
    return sum.exit_code();
  }

  auto prepared = rund::compute::pipeline(device)
                      .then(*program, rund::compute::read(*a, *b, *c),
                            rund::compute::write(*sum))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  Pipeline pipeline = std::move(prepared).value();
  const auto ran = pipeline.run();
  if (!ran) {
    return ran.exit_code();
  }
  std::array<std::int32_t, first.size()> result{};
  const auto read = pipeline.read(*sum, result);
  if (!read) {
    return read.exit_code();
  }
  return result == std::array<std::int32_t, 4u>{111, 222, 333, 444} ? 0 : 2;
}

[[nodiscard]] int Bounded(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 5u> input{5, 1, 4, 2, 3};
  auto program = rund::compute::on(device)
                     .map<std::int32_t>("pipeline-bounded", input.size(),
                                        [](auto value) { return value; })
                     .filter([](auto value) { return value > 2; })
                     .compile();
  auto source = device.upload<std::int32_t>(input);
  auto values = device.buffer<std::int32_t>(input.size());
  auto count = device.buffer<std::uint32_t>(1u);
  if (!program) {
    return program.exit_code();
  }
  if (!source) {
    return source.exit_code();
  }
  if (!values) {
    return values.exit_code();
  }
  if (!count) {
    return count.exit_code();
  }

  auto prepared = rund::compute::pipeline(device)
                      .then(*program, rund::compute::read(*source),
                            rund::compute::write(*values, *count))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  Pipeline pipeline = std::move(prepared).value();
  const auto ran = pipeline.run();
  if (!ran) {
    return ran.exit_code();
  }
  std::array<std::int32_t, input.size()> filtered{};
  std::array<std::uint32_t, 1u> logical{};
  const auto values_read = pipeline.read(*values, filtered);
  const auto count_read = pipeline.read(*count, logical);
  if (!values_read) {
    return values_read.exit_code();
  }
  if (!count_read) {
    return count_read.exit_code();
  }
  return logical[0] == 3u && filtered[0] == 5 && filtered[1] == 4 &&
                 filtered[2] == 3
             ? 0
             : 2;
}

[[nodiscard]] int RecordInSession(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> input{2, 4, 6, 8};
  auto program =
      rund::compute::on(device)
          .map<std::int32_t>("pipeline-record", input.size(),
                             [](auto value) {
                               return rund::compute::record(
                                   rund::compute::field<Value>(value),
                                   rund::compute::field<Doubled>(value * 2));
                             })
          .compile();
  auto source = device.upload<std::int32_t>(input);
  auto values = device.buffer<std::int32_t>(input.size());
  auto doubled = device.buffer<std::int32_t>(input.size());
  if (!program) {
    return program.exit_code();
  }
  if (!source) {
    return source.exit_code();
  }
  if (!values) {
    return values.exit_code();
  }
  if (!doubled) {
    return doubled.exit_code();
  }

  auto prepared = rund::compute::pipeline(device)
                      .then(*program, rund::compute::read(*source),
                            rund::compute::write(*values, *doubled))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  Pipeline pipeline = std::move(prepared).value();

  rund::Session session{};
  const auto opened = session.open(rund::SessionConfig{.workers = 2u});
  if (!opened) {
    return opened.exit_code();
  }
  auto submission = session.compute(pipeline).submit();
  const auto admitted = submission.poll();
  const auto completed = submission.wait();
  if (!admitted.submitted || admitted.reason() != rund::compute::Reason::Ok ||
      !completed) {
    (void)session.close();
    return !completed ? completed.exit_code() : 2;
  }
  const auto closed = session.close();
  if (!closed) {
    return closed.exit_code();
  }

  std::array<std::int32_t, input.size()> value_result{};
  std::array<std::int32_t, input.size()> doubled_result{};
  const auto value_read = pipeline.read(*values, value_result);
  const auto doubled_read = pipeline.read(*doubled, doubled_result);
  if (!value_read) {
    return value_read.exit_code();
  }
  if (!doubled_read) {
    return doubled_read.exit_code();
  }
  return value_result == input &&
                 doubled_result == std::array<std::int32_t, 4u>{4, 8, 12, 16}
             ? 0
             : 2;
}

[[nodiscard]] int MultiOutputAlias(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> input{3, 6, 9, 12};
  auto program = rund::compute::on(device)
                     .map<std::int32_t>("pipeline-alias", input.size(),
                                        [](auto value) { return value; })
                     .branch([](auto values) {
                       const auto doubled =
                           values.map("pipeline-alias-double",
                                      [](auto value) { return value * 2; });
                       return rund::compute::outputs(values, doubled, values);
                     })
                     .compile();
  auto source = device.upload<std::int32_t>(input);
  auto values = device.buffer<std::int32_t>(input.size());
  auto doubled = device.buffer<std::int32_t>(input.size());
  if (!program) {
    return program.exit_code();
  }
  if (!source) {
    return source.exit_code();
  }
  if (!values) {
    return values.exit_code();
  }
  if (!doubled) {
    return doubled.exit_code();
  }

  auto prepared = rund::compute::pipeline(device)
                      .then(*program, rund::compute::read(*source),
                            rund::compute::write(*values, *doubled, *values))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  Pipeline pipeline = std::move(prepared).value();
  const auto ran = pipeline.run();
  if (!ran) {
    return ran.exit_code();
  }
  std::array<std::int32_t, input.size()> value_result{};
  std::array<std::int32_t, input.size()> doubled_result{};
  const auto value_read = pipeline.read(*values, value_result);
  const auto doubled_read = pipeline.read(*doubled, doubled_result);
  if (!value_read) {
    return value_read.exit_code();
  }
  if (!doubled_read) {
    return doubled_read.exit_code();
  }
  return value_result == input &&
                 doubled_result == std::array<std::int32_t, 4u>{6, 12, 18, 24}
             ? 0
             : 2;
}

[[nodiscard]] int Recurrence(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> seed{1, 2, 3, 4};
  auto body = rund::compute::on(device)
                  .map<std::int32_t>("pipeline-repeat", seed.size(),
                                     [](auto value) { return value + 3; })
                  .compile();
  auto input = device.upload<std::int32_t>(seed);
  auto output = device.buffer<std::int32_t>(seed.size());
  auto history = device.buffer<std::int32_t>(8u * seed.size());
  if (!body || !input || !output || !history) {
    return 1;
  }
  auto prepared = rund::compute::pipeline(device)
                      .profile(rund::compute::PipelineProfile::Steps)
                      .repeat<8u>(*body, rund::compute::read(*input),
                                  rund::compute::write_final(*output))
                      .prepare();
  if (!prepared || !prepared->run()) {
    return 2;
  }
  std::array<std::int32_t, seed.size()> actual{};
  std::array<rund::compute::PipelineStepProfile, 8u> rows{};
  const auto read = prepared->read(*output, actual);
  const auto profile = prepared->profile(rows);
  if (!read || !profile ||
      actual != std::array<std::int32_t, 4u>{25, 26, 27, 28} ||
      prepared->stats().pipeline.step_count != 1u ||
      profile->written != rows.size() || profile->total != rows.size()) {
    return 3;
  }
  for (std::size_t index = 0u; index < rows.size(); ++index) {
    if (rows[index].index != 0u || rows[index].iteration != index ||
        rows[index].iteration_bound != rows.size()) {
      return 4;
    }
  }

  auto lossless = rund::compute::pipeline(device)
                      .repeat<8u>(*body, rund::compute::read(*input),
                                  rund::compute::write_each(*history))
                      .prepare();
  std::array<std::int32_t, 8u * seed.size()> each{};
  if (!lossless || !lossless->run() || !lossless->read(*history, each)) {
    return 5;
  }
  for (std::size_t iteration = 0u; iteration < 8u; ++iteration) {
    for (std::size_t element = 0u; element < seed.size(); ++element) {
      if (each[iteration * seed.size() + element] !=
          seed[element] + 3 * static_cast<std::int32_t>(iteration + 1u)) {
        return 6;
      }
    }
  }
  return 0;
}

[[nodiscard]] int HostFeedback(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 2u> seed{1, 4};
  auto body = rund::compute::on(device)
                  .map<std::int32_t>("pipeline-host-feedback", seed.size(),
                                     [](auto value) { return value + 1; })
                  .compile();
  auto input = device.upload<std::int32_t>(seed);
  auto output = device.buffer<std::int32_t>(seed.size());
  if (!body || !input || !output) {
    return 1;
  }
  auto prepared = rund::compute::pipeline(device)
                      .then(*body, rund::compute::read(*input),
                            rund::compute::write(*output))
                      .prepare();
  if (!prepared) {
    return 2;
  }
  std::array<std::int32_t, seed.size()> observed{};
  const auto status = rund::compute::host_feedback(
      *prepared, 3u,
      [&](rund::compute::HostIteration &step) noexcept
          -> rund::compute::Status {
        if (auto read = step.read(*output, observed); !read) {
          return read;
        }
        return step.has_next()
                   ? step.write(*input, std::span<const std::int32_t>{observed})
                   : rund::compute::Status::success();
      });
  return status && observed == std::array<std::int32_t, 2u>{4, 7} ? 0 : 3;
}

[[nodiscard]] int ActionFreeWindowOutput(rund::compute::Device &device) {
  constexpr std::size_t maximum = 6u;
  constexpr std::size_t tile = 2u;
  constexpr std::uint32_t sentinel = 0xA5A55A5Au;
  constexpr std::array<std::uint32_t, maximum> values{3u,  5u,  7u,
                                                      11u, 13u, 17u};
  constexpr std::array<std::uint32_t, tile> lanes{0u, 1u};
  constexpr std::array<std::uint32_t, 1u> outer_seed{19u};
  constexpr std::array<std::uint32_t, 1u> count_value{5u};
  constexpr std::array<std::uint32_t, maximum> empty{
      sentinel, sentinel, sentinel, sentinel, sentinel, sentinel};

  auto seed = rund::compute::on(device)
                  .input<std::uint32_t>(maximum)
                  .zip_input<std::uint32_t>(tile)
                  .zip_input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto source, auto lane, auto total, auto ordinal) {
                    auto current =
                        rund::compute::resident<maximum, tile>(total, ordinal);
                    auto indices = lane.combine(
                        "pipeline-installed-window-index", current.base(),
                        [](auto local, auto base) { return local + base; });
                    auto selected = source.gather(indices);
                    auto active =
                        current.count().map("pipeline-installed-window-count",
                                            [](auto value) { return value; });
                    return rund::compute::outputs(selected, active);
                  })
                  .compile();
  auto fold = rund::compute::on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(tile)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto outer, auto selected, auto count) {
                    auto enabled = selected.indices().combine(
                        "pipeline-installed-window-active", count.scalar(),
                        [](auto lane, auto active) {
                          return rund::compute::select(lane < active, 1u, 0u);
                        });
                    auto masked = selected.combine(
                        "pipeline-installed-window-mask", enabled,
                        [](auto value, auto active) {
                          return rund::compute::select(active != 0u, value, 0u);
                        });
                    auto sum = masked.reduce(rund::compute::Reduce::Sum);
                    auto next = outer.combine(
                        "pipeline-installed-window-fold", sum,
                        [](auto state, auto value) { return state + value; });
                    return rund::compute::outputs(next, selected);
                  })
                  .compile();
  auto source = device.upload<std::uint32_t>(values);
  auto lane = device.upload<std::uint32_t>(lanes);
  auto outer = device.upload<std::uint32_t>(outer_seed);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto windows = device.upload<std::uint32_t>(empty);
  if (!seed || !fold || !source || !lane || !outer || !count || !final ||
      !windows) {
    return 1;
  }

  const auto body = rund::compute::tile_repeat<0u>(*seed, *fold);
  auto builder = rund::compute::pipeline(device);
  builder.windows<maximum, tile>(body, rund::compute::window(*count),
                                 rund::compute::read(*outer, *source, *lane),
                                 rund::compute::write_final(*final),
                                 rund::compute::write_window(*windows));
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != 3u ||
      plan->inner_iteration_count != 0u ||
      plan->prepared_template_count != 6u ||
      plan->prepared_command_count != 6u) {
    return 2;
  }
  auto prepared = std::move(builder).prepare();
  if (!prepared || !prepared->run()) {
    return 3;
  }
  std::array<std::uint32_t, 1u> observed_final{};
  std::array<std::uint32_t, maximum> observed_windows{};
  if (!prepared->read(*final, observed_final) ||
      !prepared->read(*windows, observed_windows)) {
    return 4;
  }
  return observed_final[0] == 58u &&
                 observed_windows ==
                     std::array<std::uint32_t, maximum>{3u,  5u,  7u,
                                                        11u, 13u, sentinel} &&
                 prepared->stats().pipeline.executed_outer_window_count == 3u &&
                 prepared->stats().pipeline.executed_inner_iteration_count == 0u
             ? 0
             : 5;
}

[[nodiscard]] int ReusableCheckpoint(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 2u> initial{4, 9};
  auto advance =
      rund::compute::on(device)
          .map<std::int32_t>("pipeline-installed-checkpoint", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto published = device.upload<std::int32_t>(initial);
  auto pending = device.buffer<std::int32_t>(initial.size());
  if (!advance || !published || !pending) {
    return 1;
  }
  auto prepared = rund::compute::pipeline(device)
                      .state(*published, *pending)
                      .then(*advance, rund::compute::read(*published),
                            rund::compute::write(*pending))
                      .commit()
                      .prepare();
  if (!prepared) {
    return 2;
  }
  auto latest_result = prepared->latest_device_state();
  auto storage_result = prepared->snapshot_storage();
  if (!latest_result || !storage_result) {
    return 3;
  }
  LatestDeviceState latest = *latest_result;
  SnapshotStorage storage = std::move(storage_result).value();
  if (!prepared->snapshot_into(storage) || !storage.has_snapshot() ||
      storage.generation() != 0u || storage.capacity() != sizeof(initial) ||
      prepared->checkpoint_stats().device_state_acquire_count != 1u) {
    return 4;
  }
  if (!prepared->run() || latest.generation() != 1u) {
    return 5;
  }

  auto resumed = rund::compute::pipeline(device)
                     .state(*published, *pending)
                     .then(*advance, rund::compute::read(*published),
                           rund::compute::write(*pending))
                     .restore(latest)
                     .commit()
                     .prepare();
  if (!resumed || resumed->generation() != 1u ||
      resumed->checkpoint_stats().device_state_rebase_count != 1u ||
      resumed->checkpoint_stats().device_state_copy_byte_count != 0u ||
      !resumed->snapshot_into(storage) || storage.generation() != 1u) {
    return 6;
  }

  auto restored_first = device.buffer<std::int32_t>(initial.size());
  auto restored_second = device.buffer<std::int32_t>(initial.size());
  if (!restored_first || !restored_second) {
    return 7;
  }
  auto restored = rund::compute::pipeline(device)
                      .state(*restored_first, *restored_second)
                      .then(*advance, rund::compute::read(*restored_first),
                            rund::compute::write(*restored_second))
                      .restore(storage)
                      .commit()
                      .prepare();
  if (!restored || restored->generation() != 1u || !restored->run()) {
    return 8;
  }
  std::array<std::int32_t, initial.size()> observed{};
  const auto read = restored->read(*restored_second, observed);
  return read && observed == std::array<std::int32_t, 2u>{6, 11} ? 0 : 9;
}

[[nodiscard]] int NestedResidentRecurrence(rund::compute::Device &device) {
  constexpr std::size_t maximum = 5u;
  constexpr std::size_t tile = 2u;
  constexpr std::size_t inner = 3u;
  constexpr std::size_t outer_windows = (maximum + tile - 1u) / tile;
  constexpr std::size_t prepared_templates = outer_windows + 2u + 3u;
  constexpr std::size_t prepared_commands = outer_windows * (inner + 2u);
  constexpr std::array<std::uint32_t, 1u> outer_seed{10u};
  constexpr std::array<std::uint32_t, 1u> count_value{maximum};

  auto seed = rund::compute::on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto count, auto ordinal) {
                    (void)count;
                    return ordinal.map("pipeline-installed-nested-seed",
                                       [](auto value) { return value + 1u; });
                  })
                  .compile();
  auto action = rund::compute::on(device)
                    .map<std::uint32_t>("pipeline-installed-nested-action", 1u,
                                        [](auto value) { return value + 1u; })
                    .compile();
  auto fold = rund::compute::on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto outer, auto local) {
                    return outer.combine(
                        "pipeline-installed-nested-fold", local,
                        [](auto left, auto right) { return left + right; });
                  })
                  .compile();
  auto outer = device.upload<std::uint32_t>(outer_seed);
  auto count = device.upload<std::uint32_t>(count_value);
  auto output = device.buffer<std::uint32_t>(1u);
  if (!seed || !action || !fold || !outer || !count || !output) {
    return 1;
  }

  const auto body = rund::compute::tile_repeat<inner>(*seed, *action, *fold);
  auto builder = rund::compute::pipeline(device);
  builder.windows<maximum, tile>(body, rund::compute::window(*count),
                                 rund::compute::read(*outer),
                                 rund::compute::write_final(*output));
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != outer_windows ||
      plan->tile_capacity != tile || plan->inner_iteration_count != inner ||
      plan->prepared_template_count != prepared_templates ||
      plan->prepared_command_count != prepared_commands) {
    return 2;
  }

  auto prepared = std::move(builder).prepare();
  if (!prepared || !prepared->run()) {
    return 3;
  }
  std::array<std::uint32_t, 1u> actual{};
  const auto read = prepared->read(*output, actual);
  return read && actual[0] == 25u &&
                 prepared->stats().pipeline.executed_outer_window_count ==
                     outer_windows &&
                 prepared->stats().pipeline.executed_inner_iteration_count ==
                     outer_windows * inner
             ? 0
             : 4;
}

} // namespace

int main() {
  auto opened = rund::compute::open(rund::compute::Target::cpu(2u));
  if (!opened) {
    return opened.exit_code();
  }
  if (const int result = DependentWide(*opened); result != 0) {
    return result;
  }
  if (const int result = MultiInput(*opened); result != 0) {
    return result;
  }
  if (const int result = Bounded(*opened); result != 0) {
    return result;
  }
  if (const int result = RecordInSession(*opened); result != 0) {
    return result;
  }
  if (const int result = MultiOutputAlias(*opened); result != 0) {
    return result;
  }
  if (const int result = Recurrence(*opened); result != 0) {
    return result;
  }
  if (const int result = HostFeedback(*opened); result != 0) {
    return result;
  }
  if (const int result = ActionFreeWindowOutput(*opened); result != 0) {
    return result;
  }
  if (const int result = ReusableCheckpoint(*opened); result != 0) {
    return result;
  }
  if (const int result = NestedResidentRecurrence(*opened); result != 0) {
    return result;
  }
  return 0;
}
