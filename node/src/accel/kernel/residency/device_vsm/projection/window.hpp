#pragma once

#include "../projection.hpp"
#include "../source/window.hpp"

#include "../../../prepared/model.hpp"

namespace rund::node::accel::detail::device_vsm_window_projection {

struct WindowMapAuthority final {
  const BoundStep *step{};
  rund::kernel::BindingSet bindings{};
  DeviceVsmWindowMap map{};
};

struct WindowAuthority final {
  const BoundStep *step{};
  const operation::Window *active{};
  const RangeBinds *bindings{};
  WindowMapAuthority before{};
  WindowMapAuthority before_second{};
  WindowMapAuthority before_third{};
  WindowMapAuthority after{};
  WindowMapAuthority after_second{};
  WindowMapAuthority after_third{};
  DeviceVsmWindowFusion fusion{};
};

[[nodiscard]] const BoundStep *map_step(const prepared::RunState *,
                                        bool);

[[nodiscard]] bool same_parameters(const rund::kernel::BindingSet &,
                                   const rund::kernel::BindingSet &);

[[nodiscard]] bool exact_window_pipeline(const prepared::PipelineState &,
                                         WindowAuthority &,
                                         const char *&);

[[nodiscard]] bool same_window_pipeline(const prepared::PipelineState &,
                                        const WindowAuthority &);

[[nodiscard]] DeviceVsmWindowMapSources
map_sources(const WindowAuthority &) noexcept;

} // namespace rund::node::accel::detail::device_vsm_window_projection
