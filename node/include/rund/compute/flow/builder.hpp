#pragma once

#include <rund/compute/flow/bound.hpp>
#include <rund/compute/flow/bounded.hpp>
#include <rund/compute/flow/deferred.hpp>
#include <rund/compute/flow/recipe.hpp>
#include <rund/compute/flow/record.hpp>

namespace rund::compute {
class FlowBuilder final {
public:
  template <detail::ComputeValue T>
  [[nodiscard]] auto input(const std::size_t count) const {
    auto state = make(detail::type<T>(), count, detail::storage_format<T>());
    const std::array values{detail::flow_value(state)};
    return Flow<detail::InputSet<T>, stage::Exact, input::Deferred>{
        std::move(state), values};
  }

  // Re-enters a device-produced bounded value at a Program boundary.  The
  // public schema spelling stays Bounded<T, Count>; the compiled Program
  // signature is deliberately flattened to (T values, Count count), matching
  // Pipeline's ordinary typed Buffer bindings without a wrapper allocation.
  template <class Schema>
    requires detail::is_bounded<Schema>
  [[nodiscard]] auto input(const std::size_t capacity) const {
    using T = typename detail::BoundedTraits<Schema>::Value;
    using Count = typename detail::BoundedTraits<Schema>::CountType;
    static_assert(detail::ComputeValue<T>,
                  "compute bounded input value must be a compute scalar");
    static_assert(std::unsigned_integral<Count>,
                  "compute bounded input count must be unsigned");
    static_assert(sizeof(Count) == sizeof(T),
                  "compute bounded input count must match value width");
    auto state = make(detail::type<T>(), capacity, detail::storage_format<T>());
    const std::uint32_t values = detail::flow_value(state);
    const std::uint32_t count = detail::flow_independent_input(
        state, {nullptr, 1u, detail::type<Count>()}, {});
    detail::flow_mark_bounded_input(state, count, capacity);
    return Flow<T(T, Count), stage::Bounded<Count>, input::Deferred>{
        std::move(state), values, count};
  }

  template <detail::ComputeValue T, class Fn>
  [[nodiscard]] auto map(const std::string_view name, const std::size_t count,
                         Fn &&function) const {
    detail::DeferredFlow<T(T)> plan{
        make(detail::type<T>(), count, detail::storage_format<T>())};
    return std::move(plan).map(name, std::forward<Fn>(function));
  }

private:
  friend FlowBuilder on(Target) noexcept;
  friend FlowBuilder on(const Device &) noexcept;
  friend FlowBuilder on(const Device &, const ProgramCache &) noexcept;
  explicit FlowBuilder(const Target target) noexcept : target_(target) {}
  explicit FlowBuilder(std::shared_ptr<detail::DeviceState> device) noexcept
      : device_(std::move(device)), invalid_device_(device_ == nullptr) {}
  FlowBuilder(std::shared_ptr<detail::DeviceState> device,
              std::shared_ptr<detail::ProgramCacheState> cache) noexcept
      : device_(std::move(device)), cache_(std::move(cache)),
        invalid_cache_(cache_ == nullptr || device_ == nullptr) {}
  [[nodiscard]] std::shared_ptr<detail::FlowState>
  make(const detail::Type type, const std::size_t count,
       const detail::FixedFormat fixed_format = {}) const {
    auto state =
        device_ != nullptr
            ? detail::make_flow_on(device_, type, count, cache_, fixed_format)
            : detail::make_flow(target_, type, count, fixed_format);
    if (invalid_device_) {
      detail::flow_reject(state, Reason::DeviceInvalid);
    } else if (invalid_cache_) {
      detail::flow_reject(state, Reason::ProgramCacheInvalid);
    }
    return state;
  }
  Target target_{Target::cpu()};
  std::shared_ptr<detail::DeviceState> device_;
  std::shared_ptr<detail::ProgramCacheState> cache_;
  bool invalid_device_{};
  bool invalid_cache_{};
};
[[nodiscard]] inline FlowBuilder on(const Target target) noexcept {
  return FlowBuilder{target};
}
[[nodiscard]] inline FlowBuilder on(const Device &device) noexcept {
  return FlowBuilder{device.state_};
}
template <detail::ComputeValue T>
Flow<T(T)> detail::FlowFactory::make(const Target target,
                                     const std::span<const T> input) {
  auto state = make_flow(target, detail::type<T>(), input.size(),
                         detail::storage_format<T>());
  flow_bind(state, {input.data(), input.size(), detail::type<T>()});
  return Flow<T(T)>{std::move(state)};
}
template <class Range>
  requires detail::ComputeRange<Range>
[[nodiscard]] auto on(const Target target, Range &input) {
  using Value = detail::BorrowedRangeValue<Range>;
  return detail::FlowFactory::make(target, std::span<const Value>{input});
}

} // namespace rund::compute
