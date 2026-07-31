#pragma once
[[nodiscard]] inline std::string
StencilFunctionName(const rund::kernel::StencilOp op,
                    const rund::kernel::StencilElement element,
                    const rund::kernel::ComputeDomain domain) {
  std::string name = "rund_compute_stencil_";
  name += StencilOpName(op);
  name += IsSignedDomain(domain) ? "_i" : "_u";
  name += element == rund::kernel::StencilElement::U64 ? "64" : "32";
  return name;
}
