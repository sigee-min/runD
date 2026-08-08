#pragma once

#include "../kernel/scratch.hpp"
#include "state.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

class MetalScratch final {
public:
  MetalScratch(const rund::AccelDevice &pick, const KernelScratchLayout &layout,
               const RunBinds &binds) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool used() const noexcept;
  [[nodiscard]] bool active() const noexcept;
  // Maximum physical page extent.  The frozen RangeAggregate placement plan
  // uses the same page convention as Pipeline planning; this accessor lets a
  // backend re-materialize that plan without inventing another arena owner.
  [[nodiscard]] std::uint64_t page_bytes() const noexcept;
  void reset() noexcept;
  [[nodiscard]] MetalRuntimeBuffer acquire(std::uint64_t bytes) noexcept;
  [[nodiscard]] MetalRuntimeBuffer
  acquire(const KernelScratchPlacement &placement) noexcept;

private:
  struct Page final {
    std::shared_ptr<void> buffer{};
    std::uint64_t base{};
    std::uint64_t bytes{};
    std::uint64_t used{};
  };

  MetalAdapter *adapter_{};
  std::vector<Page> pages_{};
  bool valid_{};
  bool used_{};
};

class MetalScratchScope final {
public:
  explicit MetalScratchScope(MetalScratch *scratch) noexcept;
  ~MetalScratchScope();

  MetalScratchScope(const MetalScratchScope &) = delete;
  MetalScratchScope &operator=(const MetalScratchScope &) = delete;

private:
  MetalScratch *prior_{};
};

[[nodiscard]] MetalScratch *ActiveMetalScratch() noexcept;

#endif

} // namespace rund::node::accel::detail
