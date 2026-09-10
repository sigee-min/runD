#pragma once

#include "../../proof.hpp"

#include <string>

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

void append_metal_wavefront_state(std::string &);
void append_metal_wavefront_batch_begin(std::string &,
                                        const DeviceVsmGraphWavefrontProof &);
void append_metal_wavefront_batch_end(std::string &);
void append_metal_wavefront_final(std::string &);
void append_vulkan_wavefront_declarations(std::string &);
void append_vulkan_wavefront_state(std::string &);
void append_vulkan_wavefront_batch_begin(std::string &,
                                         const DeviceVsmGraphWavefrontProof &);
void append_vulkan_wavefront_batch_end(std::string &);
void append_vulkan_wavefront_final(std::string &);

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
