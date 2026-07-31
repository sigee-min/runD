#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeDef def(const char *const name) {
  return ComputeDef{name != nullptr ? std::string{name} : std::string{}};
}

[[nodiscard]] inline auto bind(const rund::kernel::u64 tile_count) {
  return detail::BindBuilder<detail::ScalarMode::Unspecified>{tile_count};
}

template <typename T> [[nodiscard]] constexpr auto buffer() noexcept {
  return detail::buffer<T>();
}

} // namespace rund::compute_dsl
