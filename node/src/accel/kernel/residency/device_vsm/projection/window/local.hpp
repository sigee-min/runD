#pragma once

#include "../window.hpp"

namespace rund::node::accel::detail::device_vsm_window_projection::
    window_detail {

[[nodiscard]] bool project_prefix_map(const prepared::RunState &,
                                      std::size_t &cursor, WindowMapAuthority &,
                                      DeviceVsmWindowMap &,
                                      const char *&reason);
[[nodiscard]] bool project_suffix_map(const prepared::RunState &,
                                      std::size_t &cursor, WindowMapAuthority &,
                                      DeviceVsmWindowMap &,
                                      const char *&reason);
[[nodiscard]] bool exact_step(const prepared::RunState *, WindowAuthority &,
                              const char *&reason);
[[nodiscard]] bool same_authority(const WindowAuthority &,
                                  const WindowAuthority &);

} // namespace
  // rund::node::accel::detail::device_vsm_window_projection::window_detail
