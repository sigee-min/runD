#include "core.hpp"

namespace rund::measure::telemetry {

[[nodiscard]] bool Identity(const std::string_view value) noexcept {
  return value.size() == 64u &&
         std::all_of(value.begin(), value.end(), [](const char byte) {
           return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
         });
}

[[nodiscard]] int Run(const std::string_view manifest,
                      const std::string_view artifact) {
  const std::optional<std::string> encoded = Seed();
  if (!encoded.has_value()) {
    std::fprintf(stderr, "telemetry expected Record preparation failed\n");
    return 1;
  }
  const auto payload = Payload();
  rund::replay::Binding replay{};
  auto source = [](rund::replay::Writer &) -> std::uint64_t {
    return kSequence;
  };
  const auto commands = replay.input(kInput, source);
  const std::array choices{commands.choice(kSequence, payload)};

  for (std::size_t warmup = 0u; warmup < kWarmups; ++warmup) {
    for (const Operation operation : OperationOrder(warmup)) {
      for (const Setting setting : SettingOrder(warmup)) {
        if (!Cold(setting, operation, *encoded, choices, false).ok) {
          std::fprintf(
              stderr, "cold telemetry warm-up failed: operation=%.*s\n",
              static_cast<int>(Name(operation).size()), Name(operation).data());
          return 1;
        }
      }
    }
  }

  Suite suite{};
  for (std::size_t pair = 0u; pair < kPairs; ++pair) {
    for (const Operation operation : OperationOrder(pair)) {
      Group &group = GroupOf(suite, Lifecycle::Cold, operation);
      for (const Setting setting : SettingOrder(pair)) {
        Pick(group, setting, pair) =
            Cold(setting, operation, *encoded, choices, true);
      }
      if (!Parity(group, pair)) {
        std::fprintf(stderr,
                     "cold telemetry parity failed: operation=%.*s pair=%zu\n",
                     static_cast<int>(Name(operation).size()),
                     Name(operation).data(), pair + 1u);
        return 1;
      }
    }
  }

  std::array<Lane, kOperations * kSettings> lanes{};
  const auto close_lanes = [&]() {
    bool ok = true;
    for (Lane &lane : lanes) {
      if (lane.opened) {
        const auto closed = lane.session.close();
        ok = closed && ok;
        lane.opened = false;
      }
    }
    return ok;
  };
  for (const Operation operation : {Operation::Live, Operation::Record,
                                    Operation::Replay, Operation::Scenario}) {
    for (const Setting setting :
         {Setting::Disabled, Setting::Basic, Setting::Detail}) {
      Lane &lane = lanes[LaneIndex(operation, setting)];
      if (NeedsExpected(operation)) {
        lane.expected = Load(*encoded);
        if (!lane.expected) {
          close_lanes();
          std::fprintf(stderr, "warm telemetry expected Record load failed\n");
          return 1;
        }
      }
      if (!lane.session.open(Config(lane.observer, setting))) {
        close_lanes();
        std::fprintf(stderr, "warm telemetry Session open failed\n");
        return 1;
      }
      lane.opened = true;
    }
  }

  for (std::size_t warmup = 0u; warmup < kWarmups; ++warmup) {
    for (const Operation operation : OperationOrder(warmup)) {
      for (const Setting setting : SettingOrder(warmup)) {
        Lane &lane = lanes[LaneIndex(operation, setting)];
        if (!Warm(lane, setting, operation, choices, warmup != 0u, false).ok) {
          close_lanes();
          std::fprintf(
              stderr,
              "warm telemetry warm-up failed: round=%zu operation=%.*s "
              "setting=%.*s expected-plan=%.*s\n",
              warmup + 1u, static_cast<int>(Name(operation).size()),
              Name(operation).data(), static_cast<int>(Name(setting).size()),
              Name(setting).data(),
              static_cast<int>(
                  Name(ExpectedPreparation(operation, warmup != 0u)).size()),
              Name(ExpectedPreparation(operation, warmup != 0u)).data());
          return 1;
        }
      }
    }
  }

  for (std::size_t pair = 0u; pair < kPairs; ++pair) {
    for (const Operation operation : OperationOrder(pair)) {
      Group &group = GroupOf(suite, Lifecycle::Warm, operation);
      for (const Setting setting : SettingOrder(pair)) {
        Lane &lane = lanes[LaneIndex(operation, setting)];
        Pick(group, setting, pair) =
            Warm(lane, setting, operation, choices, true, true);
      }
      if (!Parity(group, pair)) {
        close_lanes();
        std::fprintf(stderr,
                     "warm telemetry parity failed: operation=%.*s pair=%zu\n",
                     static_cast<int>(Name(operation).size()),
                     Name(operation).data(), pair + 1u);
        return 1;
      }
    }
  }
  if (!close_lanes()) {
    std::fprintf(stderr, "warm telemetry Session close failed\n");
    return 1;
  }

  std::printf("telemetry\t1\n");
  std::printf("identity\tmanifest\tsha256\t%.*s\n",
              static_cast<int>(manifest.size()), manifest.data());
  std::printf("identity\tartifact\tsha256\t%.*s\n",
              static_cast<int>(artifact.size()), artifact.data());
  std::printf("config\tpairs\tcount\t%zu\n", kPairs);
  std::printf("config\twarmups\tcount\t%zu\n", kWarmups);
  std::printf("config\toperations\tcount\t%zu\n", kOperations);
  std::fputs("design\toperations\torder\twilliams\n", stdout);
  std::fputs("design\tlevels\torder\tpermutations\n", stdout);
  std::fputs(
      "columns\tsample\toperation\tlifecycle\tpair\tlevel\toperation:order"
      "\tlevel:order\ttime:wall:ns\ttime:cpu:ns\tmemory:allocations"
      "\ttelemetry:detail:prepare:ns\ttelemetry:detail:work:ns"
      "\ttelemetry:detail:finish:ns\tresult:code\tresult:input:hash"
      "\tresult:transcript:hash\tresult:hash\tresult:input:rows"
      "\tresult:observation:rows\tresult:host:events"
      "\tresult:trace:records\tresult:capture:records"
      "\tresult:capture:hash\tstorage:logical:bytes"
      "\tstorage:encoded:bytes\tstorage:retained:bytes"
      "\tstorage:copied:bytes\tstorage:cached:bytes"
      "\tstorage:physical:bytes\tstorage:allocated:bytes"
      "\tstorage:reserved:bytes\tstorage:growths"
      "\tstorage:chunks\tstorage:segments\tstorage:cache:hits"
      "\tstorage:cache:misses\tstorage:cache:evictions"
      "\tcapture:retained:bytes\tcapture:retained:records"
      "\tcapture:evicted:records\tcapture:dropped:records"
      "\tpublic:input:rows"
      "\tpublic:input:bytes\tpublic:produced:rows\tpublic:choices"
      "\tpublic:evidence:rows\tpublic:evidence:bytes\tpublic:retained:bytes"
      "\tpublic:copied:bytes\tpublic:physical:bytes"
      "\tpublic:allocated:bytes\tpublic:reserved:bytes"
      "\tpublic:storage:growths\tpublic:result:hash"
      "\tevent:source\tevent:session\tevent:scope\tevent:code\tevent:mode"
      "\tevent:plan\tevent:input:rows\tevent:input:bytes"
      "\tevent:produced:rows\tevent:choices\tevent:evidence:rows"
      "\tevent:evidence:bytes\tevent:retained:bytes\tevent:copied:bytes"
      "\tevent:physical:bytes\tevent:allocated:bytes\tevent:reserved:bytes"
      "\tevent:storage:growths\tevent:result:hash\n",
      stdout);
  Print(suite);
  return 0;
}

} // namespace rund::measure::telemetry
