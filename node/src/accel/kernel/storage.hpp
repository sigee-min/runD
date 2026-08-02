#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

template <typename Step, std::size_t InlineCapacity>
struct InlineStepStorage {
  std::array<Step, InlineCapacity> inline_steps{};
  std::vector<Step> overflow_steps{};
  std::size_t expected_count = 0u;
  std::size_t step_count = 0u;
  bool ok = true;

  void reserve(const std::size_t expected) {
    if (expected > InlineCapacity) {
      overflow_steps.reserve(expected);
    } else {
      overflow_steps.clear();
    }
    expected_count = expected;
  }

  void push_back(Step&& step, const std::size_t expected) {
    if (!ok || expected != expected_count || step_count >= expected_count ||
        expected_count == 0u) {
      ok = false;
      return;
    }
    if (expected_count <= InlineCapacity) {
      inline_steps[step_count] = std::move(step);
    } else {
      overflow_steps.push_back(std::move(step));
    }
    ++step_count;
  }

  [[nodiscard]] bool valid() const noexcept {
    if (!ok || step_count != expected_count) {
      return false;
    }
    if (expected_count <= InlineCapacity) {
      return overflow_steps.empty();
    }
    return overflow_steps.size() == expected_count;
  }

  [[nodiscard]] std::size_t size() const noexcept { return step_count; }

  [[nodiscard]] const Step* get(const std::size_t index) const noexcept {
    if (index >= step_count) {
      return nullptr;
    }
    return expected_count <= InlineCapacity ? &inline_steps[index]
                                            : &overflow_steps[index];
  }

  [[nodiscard]] Step* get(const std::size_t index) noexcept {
    if (index >= step_count) {
      return nullptr;
    }
    return expected_count <= InlineCapacity ? &inline_steps[index]
                                            : &overflow_steps[index];
  }

  [[nodiscard]] const Step* data() const noexcept {
    if (expected_count == 0u) {
      return nullptr;
    }
    return expected_count <= InlineCapacity ? inline_steps.data()
                                            : overflow_steps.data();
  }
};

template <typename Value, std::size_t InlineCapacity>
struct InlineIndexedStorage {
  std::array<Value, InlineCapacity> inline_values{};
  std::vector<Value> overflow_values{};
  std::size_t expected_count = 0u;
  bool ok = true;

  void resize(const std::size_t expected) {
    if (expected > InlineCapacity) {
      overflow_values.resize(expected);
    } else {
      overflow_values.clear();
    }
    expected_count = expected;
  }

  [[nodiscard]] bool valid() const noexcept {
    if (!ok) {
      return false;
    }
    return expected_count <= InlineCapacity
               ? overflow_values.empty()
               : overflow_values.size() == expected_count;
  }

  [[nodiscard]] Value* get(const std::size_t index) noexcept {
    if (index >= size()) {
      ok = false;
      return nullptr;
    }
    return expected_count <= InlineCapacity ? &inline_values[index]
                                            : &overflow_values[index];
  }

  [[nodiscard]] const Value* get(const std::size_t index) const noexcept {
    if (index >= size()) {
      return nullptr;
    }
    return expected_count <= InlineCapacity ? &inline_values[index]
                                            : &overflow_values[index];
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return expected_count <= InlineCapacity ? expected_count
                                            : overflow_values.size();
  }

  [[nodiscard]] Value* data() noexcept {
    if (size() == 0u) {
      return nullptr;
    }
    return expected_count <= InlineCapacity ? inline_values.data()
                                            : overflow_values.data();
  }

  [[nodiscard]] const Value* data() const noexcept {
    if (size() == 0u) {
      return nullptr;
    }
    return expected_count <= InlineCapacity ? inline_values.data()
                                            : overflow_values.data();
  }
};

}  // namespace rund::node::accel::detail
