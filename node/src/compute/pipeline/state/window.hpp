#pragma once

#include "../../../accel/kernel/nested.hpp"
#include "plan.hpp"
#include <cstddef>
#include <cstdint>
#include <rund/compute/status.hpp>
#include <vector>

namespace rund::compute::detail {

struct PipelineWindow final {
  static constexpr std::uint32_t seed = 0u;
  static constexpr std::uint32_t first = 1u;
  static constexpr std::uint32_t second = 2u;

  // Exact state-wide count/bounds/final-selector authority transferred from
  // the cold plan. Runtime owner selection follows count's resource ordinal
  // through PipelineResource::partner for CPU and accelerator streams.
  PipelineWindowControl control{};
  std::size_t first_step{};
  // Leading output/input bank prefix sealed when no resident work executes.
  std::uint32_t recurrent_output_count{};
  node::accel::detail::NestedTemplateShape nested_shape{};

  [[nodiscard]] constexpr bool nested() const noexcept {
    return nested_shape.valid();
  }
};

struct PipelineWindowProgress final {
  std::uint32_t current{PipelineWindow::seed};
  bool stopped{};
};

struct AdmissionDraft;

// Only cold admission can create/change descriptors. Runtime gets const
// descriptors and a separate progress capability. Both live in one allocation;
// no descriptor copying, indirection owner or second vector is needed.
class PipelineWindows final {
public:
  PipelineWindows() = default;
  PipelineWindows(const PipelineWindows &) = delete;
  PipelineWindows &operator=(const PipelineWindows &) = delete;

  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
  [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
  [[nodiscard]] const PipelineWindow &
  operator[](std::size_t index) const noexcept {
    return entries_[index].descriptor;
  }
  [[nodiscard]] const PipelineWindow *
  find(std::size_t one_based) const noexcept {
    return one_based == 0u || one_based > size() ? nullptr
                                                 : &(*this)[one_based - 1u];
  }
  [[nodiscard]] PipelineWindowProgress &progress(std::size_t index) noexcept {
    return entries_[index].progress;
  }
  [[nodiscard]] const PipelineWindowProgress &
  progress(std::size_t index) const noexcept {
    return entries_[index].progress;
  }
  void reset_progress() noexcept {
    for (auto &entry : entries_) {
      entry.progress = {};
    }
  }
  [[nodiscard]] static constexpr std::size_t entry_bytes() noexcept {
    return sizeof(Entry);
  }
  [[nodiscard]] std::size_t retained_bytes() const noexcept {
    return entries_.capacity() * sizeof(Entry);
  }

private:
  friend Status admit_initial(AdmissionDraft &);
  friend Status admit_steps(AdmissionDraft &);
  struct Entry final {
    PipelineWindow descriptor{};
    PipelineWindowProgress progress{};
  };
  std::vector<Entry> entries_;
};

} // namespace rund::compute::detail
