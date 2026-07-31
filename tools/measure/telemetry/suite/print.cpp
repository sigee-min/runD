#include "core.hpp"

namespace rund::measure::telemetry {

void Number(const std::uint64_t value) {
  std::printf("\t%llu", static_cast<unsigned long long>(value));
}

void Text(const std::string_view value) {
  std::printf("\t%.*s", static_cast<int>(value.size()), value.data());
}

void PrintMagnitude(const Half value) {
  std::printf("%llu%s", static_cast<unsigned long long>(value.whole),
              value.half ? ".5" : "");
}

void PrintDelta(const Half value) {
  const std::string_view direction = Name(value.direction);
  std::printf("%.*s\t", static_cast<int>(direction.size()), direction.data());
  PrintMagnitude(value);
}

void PrintSample(const Operation operation, const Lifecycle lifecycle,
                 const std::size_t pair, const Setting setting,
                 const std::string_view operation_order,
                 const std::string_view level_order, const Sample &sample) {
  const Counters &counter = sample.semantics.counters;
  const ReplayFields &formula = sample.semantics.replay;
  const bool enabled = setting != Setting::Disabled;
  std::fputs("sample", stdout);
  Text(Name(operation));
  Text(Name(lifecycle));
  Number(pair + 1u);
  Text(Name(setting));
  Text(operation_order);
  Text(level_order);
  Number(sample.wall);
  Number(sample.cpu);
  Number(sample.allocations);
  Number(sample.phases.prepare_ns);
  Number(sample.phases.work_ns);
  Number(sample.phases.finish_ns);
  Number(sample.semantics.status);
  Number(sample.semantics.ordering);
  Number(sample.semantics.transcript);
  Number(sample.semantics.result);
  Number(counter.inputs);
  Number(counter.observations);
  Number(counter.host_events);
  Number(counter.trace_records);
  Number(counter.captures);
  Number(counter.capture_hash);
  Number(counter.logical_bytes);
  Number(counter.encoded_bytes);
  Number(counter.retained_bytes);
  Number(counter.copied_bytes);
  Number(counter.cached_bytes);
  Number(counter.physical_bytes);
  Number(counter.allocated_bytes);
  Number(counter.reserved_bytes);
  Number(counter.growths);
  Number(counter.chunks);
  Number(counter.segments);
  Number(counter.cache_hits);
  Number(counter.cache_misses);
  Number(counter.cache_evictions);
  Number(counter.capture_bytes);
  Number(counter.capture_records);
  Number(counter.capture_evicted);
  Number(counter.capture_dropped);
  Number(formula.input_rows);
  Number(formula.input_bytes);
  Number(formula.produced_rows);
  Number(formula.choices);
  Number(formula.evidence_rows);
  Number(formula.evidence_bytes);
  Number(formula.retained_bytes);
  Number(formula.copied_bytes);
  Number(formula.physical_bytes);
  Number(formula.allocated_bytes);
  Number(formula.reserved_bytes);
  Number(formula.storage_growths);
  Number(formula.result_hash);
  Text(enabled ? Name(sample.telemetry.source) : std::string_view{"none"});
  Number(enabled ? sample.telemetry.session : 0u);
  Number(enabled ? sample.telemetry.scope : 0u);
  Number(enabled ? sample.telemetry.code : 0u);
  Text(enabled ? Name(sample.telemetry.mode) : std::string_view{"none"});
  Text(enabled ? Name(sample.telemetry.plan) : std::string_view{"none"});
  Number(enabled ? sample.telemetry.replay.input_rows : 0u);
  Number(enabled ? sample.telemetry.replay.input_bytes : 0u);
  Number(enabled ? sample.telemetry.replay.produced_rows : 0u);
  Number(enabled ? sample.telemetry.replay.choices : 0u);
  Number(enabled ? sample.telemetry.replay.evidence_rows : 0u);
  Number(enabled ? sample.telemetry.replay.evidence_bytes : 0u);
  Number(enabled ? sample.telemetry.replay.retained_bytes : 0u);
  Number(enabled ? sample.telemetry.replay.copied_bytes : 0u);
  Number(enabled ? sample.telemetry.replay.physical_bytes : 0u);
  Number(enabled ? sample.telemetry.replay.allocated_bytes : 0u);
  Number(enabled ? sample.telemetry.replay.reserved_bytes : 0u);
  Number(enabled ? sample.telemetry.replay.storage_growths : 0u);
  Number(enabled ? sample.telemetry.replay.result_hash : 0u);
  std::putchar('\n');
}

void PrintAbsolute(const Operation operation, const Lifecycle lifecycle,
                   const std::string_view level, const std::string_view metric,
                   const std::string_view unit,
                   const std::array<std::uint64_t, kPairs> &values) {
  const Half median = Average(values[kPairs / 2u - 1u], values[kPairs / 2u]);
  constexpr std::size_t p95 = (95u * kPairs + 99u) / 100u - 1u;
  std::printf("summary\t%.*s\t%.*s\t%.*s\t%.*s\tmedian\t%.*s\tabsolute\t",
              static_cast<int>(Name(operation).size()), Name(operation).data(),
              static_cast<int>(Name(lifecycle).size()), Name(lifecycle).data(),
              static_cast<int>(level.size()), level.data(),
              static_cast<int>(metric.size()), metric.data(),
              static_cast<int>(unit.size()), unit.data());
  PrintMagnitude(median);
  std::putchar('\n');
  std::printf("summary\t%.*s\t%.*s\t%.*s\t%.*s\tp95\t%.*s\tabsolute\t%llu\n",
              static_cast<int>(Name(operation).size()), Name(operation).data(),
              static_cast<int>(Name(lifecycle).size()), Name(lifecycle).data(),
              static_cast<int>(level.size()), level.data(),
              static_cast<int>(metric.size()), metric.data(),
              static_cast<int>(unit.size()), unit.data(),
              static_cast<unsigned long long>(values[p95]));
}

void PrintDifference(const Operation operation, const Lifecycle lifecycle,
                     const std::string_view level,
                     const std::string_view metric, const std::string_view unit,
                     const std::array<Delta, kPairs> &values) {
  const Half median = Average(values[kPairs / 2u - 1u], values[kPairs / 2u]);
  constexpr std::size_t p95 = (95u * kPairs + 99u) / 100u - 1u;
  std::printf("summary\t%.*s\t%.*s\t%.*s\t%.*s\tmedian\t%.*s\t",
              static_cast<int>(Name(operation).size()), Name(operation).data(),
              static_cast<int>(Name(lifecycle).size()), Name(lifecycle).data(),
              static_cast<int>(level.size()), level.data(),
              static_cast<int>(metric.size()), metric.data(),
              static_cast<int>(unit.size()), unit.data());
  PrintDelta(median);
  std::putchar('\n');
  const std::string_view direction = Name(values[p95].direction);
  std::printf("summary\t%.*s\t%.*s\t%.*s\t%.*s\tp95\t%.*s\t%.*s\t%llu\n",
              static_cast<int>(Name(operation).size()), Name(operation).data(),
              static_cast<int>(Name(lifecycle).size()), Name(lifecycle).data(),
              static_cast<int>(level.size()), level.data(),
              static_cast<int>(metric.size()), metric.data(),
              static_cast<int>(unit.size()), unit.data(),
              static_cast<int>(direction.size()), direction.data(),
              static_cast<unsigned long long>(values[p95].magnitude));
}

void PrintSummary(const Operation operation, const Lifecycle lifecycle,
                  const Group &group) {
  for (std::size_t metric = 0u; metric < kMetrics; ++metric) {
    std::array<std::uint64_t, kPairs> disabled{};
    std::array<std::uint64_t, kPairs> basic{};
    std::array<std::uint64_t, kPairs> detail{};
    std::array<Delta, kPairs> deltas{};
    std::array<Delta, kPairs> references{};
    const std::string_view metric_name = MetricName(metric);
    for (std::size_t pair = 0u; pair < kPairs; ++pair) {
      disabled[pair] = Metric(group.disabled[pair], metric);
      basic[pair] = Metric(group.basic[pair], metric);
      detail[pair] = Metric(group.detail[pair], metric);
      deltas[pair] = Difference(basic[pair], detail[pair]);
      references[pair] = Difference(disabled[pair], basic[pair],
                                    Direction::Disabled, Direction::Basic);
      for (const auto row : {std::pair{"delta", deltas[pair]},
                             std::pair{"reference", references[pair]}}) {
        std::printf(
            "%s\t%.*s\t%.*s\t%zu\t%.*s\t%.*s\t%llu\n", row.first,
            static_cast<int>(Name(operation).size()), Name(operation).data(),
            static_cast<int>(Name(lifecycle).size()), Name(lifecycle).data(),
            pair + 1u, static_cast<int>(metric_name.size()), metric_name.data(),
            static_cast<int>(Name(row.second.direction).size()),
            Name(row.second.direction).data(),
            static_cast<unsigned long long>(row.second.magnitude));
      }
    }
    std::sort(disabled.begin(), disabled.end());
    std::sort(basic.begin(), basic.end());
    std::sort(detail.begin(), detail.end());
    std::sort(deltas.begin(), deltas.end(), Less);
    std::sort(references.begin(), references.end(), Less);
    const std::string_view unit = MetricUnit(metric);
    PrintAbsolute(operation, lifecycle, "disabled", metric_name, unit,
                  disabled);
    PrintAbsolute(operation, lifecycle, "basic", metric_name, unit, basic);
    PrintAbsolute(operation, lifecycle, "detail", metric_name, unit, detail);
    PrintDifference(operation, lifecycle, "delta", metric_name, unit, deltas);
    PrintDifference(operation, lifecycle, "reference", metric_name, unit,
                    references);
  }
}

void Print(const Suite &suite) {
  constexpr std::array operation_positions{"first", "second", "third", "last"};
  constexpr std::array level_positions{"first", "middle", "last"};
  for (const Lifecycle lifecycle : {Lifecycle::Cold, Lifecycle::Warm}) {
    for (std::size_t pair = 0u; pair < kPairs; ++pair) {
      const auto operations = OperationOrder(pair);
      const auto settings = SettingOrder(pair);
      for (std::size_t operation_position = 0u;
           operation_position < operations.size(); ++operation_position) {
        const Operation operation = operations[operation_position];
        const Group &group = GroupOf(suite, lifecycle, operation);
        for (std::size_t level_position = 0u; level_position < settings.size();
             ++level_position) {
          const Setting setting = settings[level_position];
          PrintSample(operation, lifecycle, pair, setting,
                      operation_positions[operation_position],
                      level_positions[level_position],
                      Pick(group, setting, pair));
        }
      }
    }
  }
  for (const Operation operation : {Operation::Live, Operation::Record,
                                    Operation::Replay, Operation::Scenario}) {
    for (const Lifecycle lifecycle : {Lifecycle::Cold, Lifecycle::Warm}) {
      PrintSummary(operation, lifecycle, GroupOf(suite, lifecycle, operation));
    }
  }
}


} // namespace rund::measure::telemetry
