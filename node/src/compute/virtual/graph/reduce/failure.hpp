#pragma once

#include "../../../device/residency/registry.hpp"

#include <cstdint>
#include <limits>

namespace rund::compute::detail::graph_reduce {

enum class Phase : std::uint8_t {
  None,
  Prepare,
  Supply,
  Prefix,
  Middle,
  Collective,
  Output,
  Persist,
  Recovery,
};

enum class Check : std::uint8_t {
  None,
  Prepare,
  Ticket,
  Authority,
  Select,
  Execute,
  Supply,
  Collective,
  Relocate,
  Promote,
  Activate,
  Submit,
  Wait,
  Capture,
  Close,
  Terminal,
  Persist,
  Recover,
  Quarantine,
  PrefetchPoll,
  Forecast,
  SelectShape,
  Scratch,
  SupplyStage,
  PromoteCapacity,
  PipelineIndex,
  PipelineIdentity,
  PairStart,
  ForecastShape,
  FinalShape,
  ForecastScratch,
  FinalScratch,
  PipelineNull,
  PipelineBank,
  PipelineStage,
  TerminalBind,
};

struct Fail final {
  static constexpr std::uint32_t NoStage =
      std::numeric_limits<std::uint32_t>::max();
  static constexpr std::uint64_t NoEpoch =
      std::numeric_limits<std::uint64_t>::max();

  bool present{};
  std::uint64_t epoch{NoEpoch};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint32_t stage{NoStage};
  std::uint64_t batch{NoStage};
  Phase phase{Phase::None};
  Check check{Check::None};
  Reason reason{Reason::PipelineInvalid};
  residency::CloseInfo close{};
  Phase close_phase{Phase::None};
  Check close_check{Check::None};
  bool has_close{};
};

class FailLog final {
public:
  void reset() noexcept { first_ = {}; }

  void note(const std::uint32_t stage, const std::uint64_t batch,
            const Phase phase, const Check check, const Status status,
            const residency::CloseInfo *const close = nullptr) noexcept {
    note_cred(stage, batch, phase, check, status, Fail::NoEpoch,
              close == nullptr ? 0u : close->token,
              close == nullptr ? 0u : close->generation, close);
  }

  void note(const std::uint32_t stage, const std::uint64_t batch,
            const Phase phase, const Check check, const Status status,
            const std::uint64_t epoch,
            const residency::CloseInfo *const close = nullptr) noexcept {
    note_cred(stage, batch, phase, check, status, epoch,
              close == nullptr ? 0u : close->token,
              close == nullptr ? 0u : close->generation, close);
  }

  void note_cred(const std::uint32_t stage, const std::uint64_t batch,
                 const Phase phase, const Check check, const Status status,
                 const std::uint64_t epoch, const std::uint64_t token,
                 const std::uint64_t generation,
                 const residency::CloseInfo *const close = nullptr) noexcept {
    if (status || first_.present) {
      if (!status && first_.present && close != nullptr) {
        attach(stage, batch, phase, check, epoch, *close);
      }
      return;
    }
    first_.present = true;
    first_.epoch = epoch;
    first_.stage = stage;
    first_.batch = batch;
    first_.phase = phase;
    first_.check = check;
    first_.reason = status.reason();
    first_.token = token;
    first_.generation = generation;
    if (close != nullptr) {
      if (close->token == token && close->generation == generation &&
          token != 0u && generation != 0u) {
        first_.close = *close;
        first_.close_phase = phase;
        first_.close_check = check;
        first_.has_close = true;
      }
    }
  }

  void attach(const std::uint32_t stage, const std::uint64_t batch,
              const Phase phase, const Check check, const std::uint64_t epoch,
              const residency::CloseInfo &close) noexcept {
    // Recovery has its own phase/check. Only the immutable execution
    // identity may authorize an attachment.
    if (!first_.present || first_.has_close || close.token == 0u ||
        close.generation == 0u || first_.stage != stage ||
        first_.batch != batch || first_.epoch == Fail::NoEpoch ||
        epoch != first_.epoch || first_.token != close.token ||
        first_.generation != close.generation) {
      return;
    }
    first_.close = close;
    first_.has_close = true;
    first_.close_phase = phase;
    first_.close_check = check;
  }

  [[nodiscard]] bool has() const noexcept { return first_.present; }
  [[nodiscard]] const Fail &first() const noexcept { return first_; }

private:
  Fail first_{};
};

} // namespace rund::compute::detail::graph_reduce
