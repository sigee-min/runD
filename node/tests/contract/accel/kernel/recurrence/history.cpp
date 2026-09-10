#include "local.hpp"

namespace node_accel_contract {
namespace {

[[nodiscard]] bool HistoryMarkerContract() {
  Fixture ready{ComputeApi::Metal, ComputeScalar::Lane32, false, true};
  const MapRecurrence history =
      BuildMapRecurrence(ready.entries, ready.barriers);
  if (!history.ready() || !history.writes_each_iteration()) {
    return false;
  }

  Fixture bad_pitch{ComputeApi::Metal, ComputeScalar::Lane32, false, true};
  bad_pitch.occurrences[1u].refs.inline_refs[2u].offset_bytes += 4u;
  if (BuildMapRecurrence(bad_pitch.entries, bad_pitch.barriers).state !=
      MapRecurrenceState::Invalid) {
    return false;
  }

  Fixture non_total{ComputeApi::Metal, ComputeScalar::Lane32, false, true};
  for (Occurrence &occurrence : non_total.occurrences) {
    occurrence.step.map_semantic.recurrence_total = false;
  }
  if (BuildMapRecurrence(non_total.entries, non_total.barriers).state !=
      MapRecurrenceState::Ineligible) {
    return false;
  }

  Fixture mixed_marker{ComputeApi::Metal, ComputeScalar::Lane32, false, true};
  mixed_marker.entries[1u].recurrence.writes_each_iteration = false;
  return BuildMapRecurrence(mixed_marker.entries, mixed_marker.barriers)
             .state == MapRecurrenceState::Ineligible;
}

} // namespace

bool MapRecurrenceHistoryContract() {
  return HistoryMarkerContract();
}

} // namespace node_accel_contract
