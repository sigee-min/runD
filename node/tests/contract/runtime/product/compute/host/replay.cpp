#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/replay.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace runtime_compute_host_detail {

[[nodiscard]] bool CheckServerReplay(rund::Session &server) {
  using namespace rund::compute;
  constexpr std::size_t count = 4u;
  auto device = open(server, Target::cpu(2u));
  if (!device) {
    std::fprintf(stderr, "server replay device: %.*s\n",
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return false;
  }
  const auto backend = device->backend();
  if (!backend || *backend != Backend::Cpu) {
    std::fprintf(stderr, "server replay backend invalid\n");
    return false;
  }
  auto program =
      on(*device)
          .map<std::uint32_t>("server-native-map", count,
                              [](auto value) { return value * 3u + 1u; })
          .compile();
  if (!program) {
    std::fprintf(stderr, "server replay program: %.*s\n",
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  constexpr std::array<std::uint32_t, count> empty{};
  auto job = program->resident(empty);
  if (!job) {
    std::fprintf(stderr, "server replay job: %.*s\n",
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }

  std::array canonical{std::byte{1u}, std::byte{2u}, std::byte{3u},
                       std::byte{4u}};
  std::uint64_t source_calls = 0u;
  std::uint64_t callback_calls = 0u;
  bool callback_ok = true;
  std::vector<std::uint32_t> recorded{};
  std::vector<std::uint32_t> replayed{};
  rund::replay::Binding replay{};
  auto source = [&](rund::replay::Writer &writer) -> std::uint64_t {
    ++source_calls;
    if (!writer.append(canonical)) {
      callback_ok = false;
    }
    return 19u;
  };
  const auto input =
      replay.input(rund::replay::Input{.id = 71u, .schema = 7001u}, source);
  const auto run = [&](rund::replay::Context &context, rund::Session &active) {
    ++callback_calls;
    const rund::replay::Value value = input.read(context);
    if (&active != &server || !value || value.sequence() != 19u ||
        value.size() != count) {
      callback_ok = false;
      return;
    }
    std::array<std::uint32_t, count> values{};
    for (std::size_t index = 0u; index < count; ++index) {
      values[index] = std::to_integer<std::uint32_t>(value.bytes()[index]);
    }
    if (!job->write(values) || !active.compute(*job).submit().wait()) {
      callback_ok = false;
      return;
    }
    auto output = job->read();
    if (!output) {
      callback_ok = false;
      return;
    }
    (callback_calls == 1u ? recorded : replayed) = std::move(*output);
  };

  const rund::replay::Record baseline = rund::replay::record(server, run);
  if (!baseline || !callback_ok || source_calls != 1u || callback_calls != 1u) {
    std::fprintf(stderr,
                 "server replay record: %.*s callback_ok=%u source=%llu "
                 "callback=%llu\n",
                 static_cast<int>(baseline.error().size()),
                 baseline.error().data(), callback_ok ? 1u : 0u,
                 static_cast<unsigned long long>(source_calls),
                 static_cast<unsigned long long>(callback_calls));
    return false;
  }
  canonical.fill(std::byte{0xffu});
  const rund::replay::Check checked = rund::replay::run(server, baseline, run);
  constexpr std::array<std::uint32_t, count> expected{4u, 7u, 10u, 13u};
  const bool ok = checked && checked.actual_hash() == baseline.hash() &&
                  callback_ok && source_calls == 1u && callback_calls == 2u &&
                  std::ranges::equal(recorded, expected) &&
                  std::ranges::equal(replayed, expected);
  if (!ok) {
    std::fprintf(stderr,
                 "server native replay: %.*s record=%llu actual=%llu "
                 "source_calls=%llu callback_calls=%llu\n",
                 static_cast<int>(checked.error().size()),
                 checked.error().data(),
                 static_cast<unsigned long long>(baseline.hash()),
                 static_cast<unsigned long long>(checked.actual_hash()),
                 static_cast<unsigned long long>(source_calls),
                 static_cast<unsigned long long>(callback_calls));
  }
  return ok;
}

} // namespace runtime_compute_host_detail
