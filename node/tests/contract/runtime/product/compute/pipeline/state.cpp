#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace runtime_compute_pipeline {

int Publish(rund::Session &, rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> initial{1, 2, 3, 4};
  auto advance =
      on(device)
          .map<std::int32_t>("pipeline-state-advance", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto first = device.upload<std::int32_t>(initial);
  auto second = device.buffer<std::int32_t>(initial.size());
  if (!advance || !first || !second) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .state(*first, *second)
                      .then(*advance, read(*first), write(*second))
                      .commit()
                      .prepare();
  if (!prepared || prepared->generation() != 0u) {
    return 2;
  }
  std::array<std::int32_t, initial.size()> observed{};
  if (!ReadExact(*prepared, *first, std::span<std::int32_t>{observed}) ||
      observed != initial) {
    return 3;
  }
  if (!prepared->run() || prepared->generation() != 1u ||
      !ReadExact(*prepared, *second, std::span<std::int32_t>{observed}) ||
      observed != std::array<std::int32_t, 4u>{2, 3, 4, 5}) {
    return 4;
  }
  if (!prepared->run() || prepared->generation() != 2u ||
      !ReadExact(*prepared, *second, std::span<std::int32_t>{observed}) ||
      observed != std::array<std::int32_t, 4u>{3, 4, 5, 6}) {
    return 5;
  }
  auto saved = prepared->snapshot();
  if (!saved || saved->generation() != 2u ||
      saved->fingerprint() != prepared->fingerprint() || saved->hash() == 0u) {
    return 6;
  }
  const StateSnapshot copied = *saved;
  const Stats published = prepared->stats();
  if (published.publication.generation != 2u ||
      published.publication.commit_count != 2u ||
      published.publication.snapshot_byte_count != sizeof(initial) ||
      published.publication.snapshot_hash != copied.hash() ||
      copied.hash() != saved->hash()) {
    return 7;
  }

  auto replacement_device = open(Target::cpu(2u));
  if (!replacement_device) {
    return 8;
  }
  auto replacement_advance =
      on(*replacement_device)
          .map<std::int32_t>("pipeline-state-advance", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto replacement_first =
      replacement_device->buffer<std::int32_t>(initial.size());
  auto replacement_second =
      replacement_device->buffer<std::int32_t>(initial.size());
  if (!replacement_advance || !replacement_first || !replacement_second) {
    return 9;
  }
  auto rejected_restore_order =
      pipeline(*replacement_device)
          .state(*replacement_first, *replacement_second)
          .then(*replacement_advance, read(*replacement_first),
                write(*replacement_second))
          .commit()
          .restore(copied)
          .prepare();
  if (rejected_restore_order ||
      rejected_restore_order.error() != "compute_pipeline_invalid") {
    return 10;
  }
  auto restored = pipeline(*replacement_device)
                      .state(*replacement_first, *replacement_second)
                      .then(*replacement_advance, read(*replacement_first),
                            write(*replacement_second))
                      .restore(copied)
                      .profile(PipelineProfile::Steps)
                      .commit()
                      .prepare();
  if (!restored || restored->generation() != 2u ||
      restored->fingerprint() != copied.fingerprint() || copied.hash() == 0u ||
      !restored->run() || restored->generation() != 3u ||
      !ReadExact(*restored, *replacement_second,
                 std::span<std::int32_t>{observed}) ||
      observed != std::array<std::int32_t, 4u>{4, 5, 6, 7} ||
      restored->stats().publication.restore_byte_count !=
          sizeof(initial) * 2u) {
    return 11;
  }
  return 0;
}

int Failure(rund::Session &session, rund::compute::Device &device) {
  using namespace rund::compute;
  using Real = Fixed<20, 44>;
  constexpr auto scalar = [](const std::int64_t value) {
    return Real::from_raw(value << Real::fraction_bits);
  };
  constexpr std::array<Real, 4u> singular{scalar(1), scalar(2), scalar(2),
                                          scalar(4)};
  auto factor = on(device)
                    .map<Real>("pipeline-state-failure", singular.size(),
                               [](auto value) { return quantize<Real>(value); })
                    .matrix<2u, 2u>()
                    .lu()
                    .compile();
  auto published = device.upload<Real>(singular);
  auto pending = device.buffer<Real>(singular.size());
  auto pivots = device.buffer<std::uint32_t>(2u);
  auto status = device.buffer<std::uint32_t>(1u);
  if (!factor || !published || !pending || !pivots || !status) {
    return 1;
  }
  auto prepared =
      pipeline(device)
          .state(*published, *pending)
          .then(*factor, read(*published), write(*pending, *pivots, *status))
          .commit()
          .prepare();
  if (!prepared) {
    return 2;
  }
  std::array<Real, singular.size()> before{};
  if (!ReadExact(*prepared, *published, std::span<Real>{before}) ||
      before != singular) {
    return 3;
  }
  const Completion failed = session.compute(*prepared).submit().wait();
  std::array<Real, singular.size()> after{};
  auto saved = prepared->snapshot();
  if (failed || failed.reason() != Reason::FactorSingular ||
      prepared->generation() != 0u || !prepared->poisoned() ||
      !ReadExact(*prepared, *published, std::span<Real>{after}) ||
      after != before || !saved || saved->generation() != 0u ||
      prepared->stats().publication.discard_count != 1u ||
      prepared->stats().publication.commit_count != 0u) {
    return 4;
  }
  return 0;
}

int StateCancel(rund::Session &session, rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t count = 1u << 20u;
  std::vector<std::uint32_t> initial(count, 3u);
  auto advance = on(device)
                     .map<std::uint32_t>("pipeline-state-cancel", count,
                                         [](auto value) { return value + 1u; })
                     .map("pipeline-state-cancel-second",
                          [](auto value) { return value * 7u; })
                     .compile();
  auto published = device.upload<std::uint32_t>(initial);
  auto pending = device.buffer<std::uint32_t>(count);
  if (!advance || !published || !pending) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .state(*published, *pending)
                      .then(*advance, read(*published), write(*pending))
                      .commit()
                      .prepare();
  if (!prepared) {
    return 2;
  }
  constexpr std::size_t attempts = 64u;
  std::uint32_t expected = 3u;
  std::uint64_t completed_runs = 0u;
  for (std::size_t attempt = 0u; attempt < attempts; ++attempt) {
    auto submission = session.compute(*prepared).submit();
    Poll observed = submission.poll();
    for (std::size_t spin = 0u;
         spin < 1000000u && !observed.backend_submitted && !observed.completed;
         ++spin) {
      std::this_thread::yield();
      observed = submission.poll();
    }
    if (!observed.backend_submitted) {
      return 3;
    }
    if (observed.completed) {
      if (!submission.wait()) {
        return 3;
      }
      expected = (expected + 1u) * 7u;
      ++completed_runs;
      continue;
    }

    const Status cancelled = submission.cancel();
    const Completion completion = submission.wait();
    if (!cancelled) {
      if (cancelled.reason() == Reason::AlreadyCompleted && completion) {
        expected = (expected + 1u) * 7u;
        ++completed_runs;
        continue;
      }
      return 4;
    }
    auto saved = prepared->snapshot();
    std::vector<std::uint32_t> observed_values(count);
    const Status read = prepared->read(*published, observed_values);
    if (completion || completion.reason() != Reason::Cancelled ||
        prepared->poisoned() || prepared->generation() != completed_runs ||
        !saved || saved->generation() != completed_runs || !read ||
        observed_values != std::vector<std::uint32_t>(count, expected) ||
        prepared->stats().publication.discard_count != 1u ||
        prepared->stats().publication.commit_count != completed_runs) {
      return 4;
    }
    const Status retried = prepared->run();
    expected = (expected + 1u) * 7u;
    std::vector<std::uint32_t> advanced(count);
    const Status retried_read = prepared->read(*pending, advanced);
    if (!retried || prepared->generation() != completed_runs + 1u ||
        !retried_read ||
        advanced != std::vector<std::uint32_t>(count, expected) ||
        prepared->stats().publication.commit_count != completed_runs + 1u ||
        prepared->stats().publication.discard_count != 1u) {
      return 5;
    }
    return 0;
  }
  return 6;
}

} // namespace runtime_compute_pipeline
