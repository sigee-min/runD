#include "local.hpp"

#include "../exception.hpp"
#include "../input/plan.hpp"
#include <kernel/core/checked.hpp>

#include <rund/host/hash.hpp>

#include <algorithm>
#include <compare>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace rund::replay::detail {
namespace {

struct Key final {
  std::uint64_t source = 0u;
  std::uint64_t schema = 0u;
  std::uint64_t sequence = 0u;

  auto operator<=>(const Key &) const noexcept = default;
};

[[nodiscard]] Key key(const Choice &choice) noexcept {
  return Key{choice.input().id, choice.input().schema, choice.sequence()};
}

} // namespace

ReplayScenarioPlan
prepare_scenario(Session &session, const Record &expected,
                 const std::span<const Choice> choices) noexcept {
  if (const Code code = ready_code(expected); code != Code::Ok) {
    return ReplayScenarioPlan{.code = code};
  }
  if (!expected.data_->prepared) {
    return ReplayScenarioPlan{.code = Code::RecordNotPrepared};
  }
  try {
    std::size_t packed_size = 0u;
    std::uint64_t replaced_bytes = 0u;
    std::uint64_t replacement_bytes = 0u;
    bool byte_count_overflow = false;
    for (const Choice &choice : choices) {
      if (choice.input().id == 0u) {
        return ReplayScenarioPlan{.code = Code::InputIdInvalid};
      }
      if (choice.input().schema == 0u) {
        return ReplayScenarioPlan{.code = Code::InputSchemaInvalid};
      }
      const Key identity = key(choice);
      const scope::Match match = expected.data_->prepared->index().find(
          scope::Key{identity.source, identity.schema, identity.sequence});
      if (match.count == 0u) {
        return ReplayScenarioPlan{.code = Code::ScenarioInputMissing};
      }
      if (!match.unique()) {
        return ReplayScenarioPlan{.code = Code::ScenarioInputAmbiguous};
      }
      const std::uint64_t replacement =
          static_cast<std::uint64_t>(choice.bytes().size());
      std::uint64_t next = 0u;
      if (!rund::kernel::checked::add(replaced_bytes, match.bytes, next)) {
        byte_count_overflow = true;
      } else {
        replaced_bytes = next;
      }
      if (!rund::kernel::checked::add(replacement_bytes, replacement, next)) {
        byte_count_overflow = true;
      } else {
        replacement_bytes = next;
      }
      if (choice.bytes().size() >
          std::numeric_limits<std::size_t>::max() - packed_size) {
        return ReplayScenarioPlan{.code = Code::ScenarioInputCapacityExceeded};
      }
      packed_size += choice.bytes().size();
    }

    std::byte *packed = nullptr;
    node::replay_detail::payload::Bytes bytes =
        node::replay_detail::payload::Bytes::create(packed_size, packed);
    std::vector<node::replay_detail::InputPatch> patches{};
    patches.reserve(choices.size());
    std::size_t offset = 0u;
    for (const Choice &choice : choices) {
      const StableHash hash =
          host::hash_bytes(choice.bytes().data(), choice.bytes().size());
      if (!choice.bytes().empty()) {
        std::copy(choice.bytes().begin(), choice.bytes().end(),
                  packed + offset);
      }
      patches.push_back(node::replay_detail::InputPatch{
          .source = choice.input().id,
          .schema = choice.input().schema,
          .sequence = choice.sequence(),
          .payload_hash = hash.value,
          .offset = offset,
          .size = choice.bytes().size(),
      });
      offset += choice.bytes().size();
    }
    std::sort(patches.begin(), patches.end(),
              [](const node::replay_detail::InputPatch &left,
                 const node::replay_detail::InputPatch &right) {
                if (left.source != right.source) {
                  return left.source < right.source;
                }
                if (left.schema != right.schema) {
                  return left.schema < right.schema;
                }
                return left.sequence < right.sequence;
              });
    for (std::size_t index = 1u; index < patches.size(); ++index) {
      const node::replay_detail::InputPatch &previous = patches[index - 1u];
      const node::replay_detail::InputPatch &current = patches[index];
      if (previous.source == current.source &&
          previous.schema == current.schema &&
          previous.sequence == current.sequence) {
        return ReplayScenarioPlan{.code = Code::ScenarioInputDuplicate};
      }
    }

    const std::uint64_t baseline_bytes =
        expected.data_->prepared->payloads().logical_bytes();
    std::uint64_t planned_bytes = 0u;
    if (byte_count_overflow || replaced_bytes > baseline_bytes ||
        !rund::kernel::checked::add(baseline_bytes - replaced_bytes,
                                    replacement_bytes, planned_bytes)) {
      return ReplayScenarioPlan{.code = Code::ScenarioInputCapacityExceeded};
    }
    if (planned_bytes > scope::Access::capacity(session)) {
      return ReplayScenarioPlan{.code = Code::ScenarioInputCapacityExceeded};
    }
    auto frozen = std::make_shared<const node::replay_detail::InputPlan>(
        std::move(bytes), std::move(patches));
    return ReplayScenarioPlan{
        .choices = std::static_pointer_cast<const void>(std::move(frozen)),
        .code = Code::Ok,
        .bytes = planned_bytes,
    };
  } catch (...) {
    return ReplayScenarioPlan{
        .code = node::replay_detail::CurrentExceptionCode({
            .bad_alloc = Code::AllocationFailed,
            .length_error = Code::ScenarioInputCapacityExceeded,
            .unexpected = Code::ScenarioPrepareFailed,
        })};
  }
}

Scenario finish_scenario(const Record &expected, Session &session,
                         Session::Result &&actual, const bool callback_ran,
                         const std::uint64_t start_hash) noexcept {
  std::optional<Record> actual_record{
      std::in_place, build_record(session, std::move(actual), start_hash)};
  std::optional<Diff> comparison{
      std::in_place, ::rund::replay::diff(expected, *actual_record)};
  const Code code = !callback_ran          ? Code::ScenarioCallbackNotRun
                    : !actual_record->ok() ? actual_record->code()
                                           : Code::Ok;
  return Scenario{code, std::move(actual_record), std::move(comparison),
                  callback_ran};
}

Scenario fail_scenario(const Code code) noexcept {
  return Scenario{code == Code::Ok ? Code::ScenarioNotPrepared : code,
                  std::nullopt, std::nullopt, false};
}

} // namespace rund::replay::detail
