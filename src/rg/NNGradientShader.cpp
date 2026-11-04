//
// Created by adamyuan on 3/5/24.
//

#include "NNGradientShader.hpp"
#if defined(__ANDROID__) && !defined(__x86_64__)
#include "KhrMatrixShader.hpp"
#endif

#include <spdlog/spdlog.h>

namespace rg {

std::tuple<myvk::Ptr<myvk::ShaderModule>, VkPipelineShaderStageRequiredSubgroupSizeCreateInfo>
NNGradientShader::Create(const myvk::Ptr<myvk::Device> &device) {
#if defined(__ANDROID__) && defined(__x86_64__)
	static constexpr uint32_t scalar[] = {
#include <shader/nrc_gradient_scalar.comp.u32>
	};
	return {myvk::ShaderModule::Create(device, scalar, sizeof(scalar)), VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{}};
#elif defined(__ANDROID__)
	static constexpr uint32_t f16[] = {
#include <shader/nrc_gradient_khr_f16.comp.u32>
	};
	static constexpr uint32_t f32[] = {
#include <shader/nrc_gradient_khr_f32.comp.u32>
	};
	return {CreateKhrMatrixShader(device, f16, f32), VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{}};
#else
	auto subgroup_size = device->GetPhysicalDevicePtr()->GetProperties().vk11.subgroupSize;
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo info{
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
	    .requiredSubgroupSize = subgroup_size};
	if (subgroup_size == 16) {
		static constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_gradient_16.comp.u32>

		};
		return {myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv)), info};
	} else if (subgroup_size == 32) {
		static constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_gradient_32.comp.u32>

		};
		return {myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv)), info};
	} /* else if (subgroup_size == 64) {
	    static constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_gradient_64.comp.u32>
	    };
	    return {myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv)), info};
	} */
	spdlog::error("Unsupported subgroup size {}, only 16, 32, 64 are supported", subgroup_size);
	return {nullptr, info};
#endif
}

} // namespace rg
