#pragma once

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/dsl/support.hpp>

#include <memory>
#include <vector>

namespace rund::compute_dsl {

namespace detail {

struct ComputeOpMetadataView final {
  const rund::kernel::u8 *param_data = nullptr;
  rund::kernel::u64 param_bytes = 0u;
  const rund::kernel::u64 *input_element_bytes = nullptr;
  rund::kernel::u64 input_count = 0u;
  const rund::kernel::u64 *output_element_bytes = nullptr;
  rund::kernel::u64 output_count = 0u;
};

} // namespace detail

class ComputeOp {
public:
  ComputeOp() = default;
  ComputeOp(rund::kernel::ComputeIR ir, rund::kernel::ComputeMap map,
            std::vector<detail::BindingRuntime> bindings,
            rund::kernel::u64 tile_count);

  [[nodiscard]] rund::kernel::ComputeMap map() const noexcept;
  [[nodiscard]] const rund::kernel::ComputeIR &ir() const noexcept;
  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] const char *reason() const noexcept;
  [[nodiscard]] rund::kernel::InputShapeBytes
  input_element_bytes() const noexcept;
  [[nodiscard]] detail::ComputeOpMetadataView metadata_view() const noexcept;

  template <typename Output>
  [[nodiscard]] rund::kernel::BindingSet
  bindings(const rund::kernel::u64 phase_id, const rund::kernel::u64 lane_count,
           const rund::kernel::ComputeApi api) const {
    return staged(phase_id, lane_count, api);
  }

  template <typename Output>
  [[nodiscard]] rund::kernel::BindingSet
  resident_bindings(const rund::kernel::u64 phase_id,
                    const rund::kernel::u64 lane_count,
                    const rund::kernel::ComputeApi api,
                    const rund::kernel::ResidentBufferRef *const inputs,
                    const rund::kernel::u64 input_count,
                    const rund::kernel::ResidentBufferRef &output) const {
    return resident(phase_id, lane_count, api, inputs, input_count, output,
                    nullptr, 0u, nullptr);
  }

  template <typename Output>
  [[nodiscard]] rund::kernel::BindingSet
  resident_bindings(const rund::kernel::u64 phase_id,
                    const rund::kernel::u64 lane_count,
                    const rund::kernel::ComputeApi api,
                    const rund::kernel::ResidentBufferRef *const inputs,
                    const rund::kernel::u64 input_count,
                    const rund::kernel::ResidentBufferRef &output,
                    const std::shared_ptr<void> *const input_handles,
                    const rund::kernel::u64 input_handle_count,
                    const std::shared_ptr<void> *const output_handle) const {
    return resident(phase_id, lane_count, api, inputs, input_count, output,
                    input_handles, input_handle_count, output_handle);
  }

private:
  [[nodiscard]] rund::kernel::BindingSet
  staged(rund::kernel::u64 phase_id, rund::kernel::u64 lane_count,
         rund::kernel::ComputeApi api) const;
  [[nodiscard]] rund::kernel::BindingSet
  resident(rund::kernel::u64 phase_id, rund::kernel::u64 lane_count,
           rund::kernel::ComputeApi api,
           const rund::kernel::ResidentBufferRef *inputs,
           rund::kernel::u64 input_count,
           const rund::kernel::ResidentBufferRef &output,
           const std::shared_ptr<void> *input_handles,
           rund::kernel::u64 input_handle_count,
           const std::shared_ptr<void> *output_handle) const;

  rund::kernel::ComputeIR ir_{};
  rund::kernel::ComputeMap map_{};
  std::vector<detail::BindingRuntime> bindings_{};
  rund::kernel::u64 tile_count_ = 0u;
  std::vector<rund::kernel::u8> param_storage_{};
  std::vector<rund::kernel::BufferSpan> input_spans_{};
  std::vector<rund::kernel::u64> input_element_bytes_{};
  std::vector<rund::kernel::OutputSpan> output_spans_{};
  std::vector<rund::kernel::u64> output_element_bytes_{};
};

} // namespace rund::compute_dsl
