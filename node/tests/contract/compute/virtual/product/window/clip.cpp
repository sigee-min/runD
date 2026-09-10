#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace rund_node_test_virtual::product::window_detail {

int CheckClipWindow(const rund::compute::Backend backend,
                    rund::compute::Device &device, const WindowBuffer &input,
                    const std::span<std::byte> observed) {
  using namespace rund::compute;
  auto clip_flow = on(device).input<std::int32_t>(FrameElements);
  auto clip_program =
      std::move(clip_flow)
          .branch([](auto values) {
            return values.window(WindowSpec{
                .op = Window::Min, .radius = Radius, .edge = WindowEdge::Clip});
          })
          .compile();
  auto clip_backing = std::make_shared<MemoryVirtualBacking>(
      WindowElements * sizeof(std::int32_t),
      FrameElements * sizeof(std::int32_t));
  auto clip_output = virtual_buffer<std::int32_t>(WindowElements, clip_backing);
  auto mapped_clip_program =
      on(device)
          .map<std::int32_t>("virtual-product-window-clip-mapped",
                             FrameElements, [](auto item) { return item + 3; })
          .window(WindowSpec{
              .op = Window::Min, .radius = Radius, .edge = WindowEdge::Clip})
          .compile();
  auto mapped_clip_backing = std::make_shared<MemoryVirtualBacking>(
      WindowElements * sizeof(std::int32_t),
      FrameElements * sizeof(std::int32_t));
  auto mapped_clip_output =
      virtual_buffer<std::int32_t>(WindowElements, mapped_clip_backing);
  auto mapped_clip =
      mapped_clip_program && mapped_clip_output
          ? virtual_pipeline(*mapped_clip_program, input, *mapped_clip_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  const bool mapped_clip_cpu_rejected =
      backend == Backend::Cpu && !mapped_clip &&
      mapped_clip.reason() == Reason::PrimitiveUnsupported;
  const Status mapped_clip_accel = backend != Backend::Cpu && mapped_clip
                                       ? mapped_clip->run()
                                       : Status::fail(Reason::PipelineInvalid);
  const bool mapped_clip_accel_rejected =
      backend != Backend::Cpu && mapped_clip && !mapped_clip_accel &&
      mapped_clip_accel.reason() == Reason::BackendUnsupported;
  if (!mapped_clip_program ||
      (!mapped_clip_cpu_rejected && !mapped_clip_accel_rejected)) {
    std::fprintf(stderr,
                 "virtual mapped Clip prepare backend=%u program=%u "
                 "reason=%.*s run=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(static_cast<bool>(mapped_clip_program)),
                 static_cast<int>(mapped_clip.error().size()),
                 mapped_clip.error().data(),
                 static_cast<int>(mapped_clip_accel.error().size()),
                 mapped_clip_accel.error().data());
    return 8;
  }
  auto clip = clip_program && clip_output
                  ? virtual_pipeline(*clip_program, input, *clip_output,
                                     ResidencyConfig{})
                  : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                        Reason::PipelineInvalid);
  if (!clip || !clip->run() || !clip_backing->observe(observed)) {
    return 9;
  }
  for (std::size_t index = 0u; index < WindowElements; ++index) {
    std::int32_t actual = 0;
    std::memcpy(&actual, observed.data() + index * sizeof(actual),
                sizeof(actual));
    if (actual != expected_clip_min(index)) {
      return 10;
    }
  }
  if (clip->stats().pipeline.residency.page_in_count != WindowPages ||
      clip->stats().pipeline.residency.cache_hit_count != 0u) {
    return 11;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::window_detail
