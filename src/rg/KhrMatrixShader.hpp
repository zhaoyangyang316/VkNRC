#pragma once

#include "../android/AndroidMatrix.hpp"
#include "KhrMatrixShape.hpp"
#include <myvk/ShaderModule.hpp>
#include <span>
#include <stdexcept>

namespace rg {
inline myvk::Ptr<myvk::ShaderModule> CreateKhrMatrixShader(const myvk::Ptr<myvk::Device> &device,
                                                           std::span<const uint32_t> f16,
                                                           std::span<const uint32_t> f32) {
	const auto config = GetAndroidMatrixConfig(device->GetPhysicalDevicePtr()->GetHandle());
	if (!SupportsKhrMatrixShape(config.m, config.n, config.k))
		throw std::runtime_error("No compatible KHR cooperative matrix shape for NRC");
	const auto code = config.fp32_accumulator ? f32 : f16;
	auto shader = myvk::ShaderModule::Create(device, code.data(), code.size_bytes());
	if (!shader)
		throw std::runtime_error("Cannot create KHR cooperative matrix shader module");
	shader->AddSpecialization(1, config.m);
	shader->AddSpecialization(2, config.n);
	shader->AddSpecialization(3, config.k);
	return shader;
}
} // namespace rg
