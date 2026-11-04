#include "AndroidDevice.hpp"
#include "AndroidMatrix.hpp"
#include "../rg/KhrMatrixShape.hpp"

#include <optional>
#include <sstream>

namespace {
const std::vector<const char *> kExtensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};

VkNRCMatrixConfig SelectMatrix(VkPhysicalDevice device, std::ostream *report) {
	if (!vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
		return {};
	std::vector<VkCooperativeMatrixPropertiesKHR> tiles;
	uint32_t count{};
	VkResult result;
	do {
		if (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &count, nullptr) != VK_SUCCESS || !count)
			return {};
		tiles.assign(count, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
		result = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &count, tiles.data());
	} while (result == VK_INCOMPLETE);
	if (result != VK_SUCCESS)
		return {};
	tiles.resize(count);
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(device, &properties);
	VkNRCMatrixConfig best{};
	for (const auto &t : tiles) {
		if (report)
			*report << "  tile " << t.MSize << 'x' << t.NSize << 'x' << t.KSize
			        << " A/B/C/result=" << t.AType << '/' << t.BType << '/' << t.CType << '/' << t.ResultType
			        << " scope=" << t.scope << " saturating=" << t.saturatingAccumulation << '\n';
		bool fp32 = t.CType == VK_COMPONENT_TYPE_FLOAT32_KHR;
		if (t.scope != VK_SCOPE_SUBGROUP_KHR || t.saturatingAccumulation || t.AType != VK_COMPONENT_TYPE_FLOAT16_KHR ||
		    t.BType != t.AType || t.ResultType != t.CType || (!fp32 && t.CType != t.AType) ||
		    !SupportsKhrMatrixShape(t.MSize, t.NSize, t.KSize) ||
		    KhrMatrixSharedBytes(true, fp32) > properties.limits.maxComputeSharedMemorySize)
			continue;
		if (!best.m || (best.fp32_accumulator == fp32 && t.MSize*t.NSize*t.KSize > best.m*best.n*best.k) ||
		    (best.fp32_accumulator && !fp32))
			best = {t.MSize, t.NSize, t.KSize, fp32};
	}
	return best;
}
} // namespace

VkNRCMatrixConfig GetAndroidMatrixConfig(VkPhysicalDevice device) { return SelectMatrix(device, nullptr); }

myvk::Ptr<myvk::Device> CreateAndroidDevice(const myvk::Ptr<myvk::Instance> &instance, VkSurfaceKHR surface,
                                            myvk::Ptr<myvk::Queue> &queue, std::string &report,
                                            bool reference) {
	// Match the packaged ABI's shader set. The phone must run real KHR matrices.
	const std::vector<const char *> extensions = reference ? std::vector<const char *>{VK_KHR_SWAPCHAIN_EXTENSION_NAME} : kExtensions;
	std::ostringstream log;
	for (const auto &gpu : myvk::PhysicalDevice::Fetch(instance)) {
		const auto &p = gpu->GetProperties();
		const auto &f = gpu->GetFeatures();
		const auto handle = gpu->GetHandle();
		log << p.vk10.deviceName << "  Vulkan " << VK_VERSION_MAJOR(p.vk10.apiVersion) << '.'
		    << VK_VERSION_MINOR(p.vk10.apiVersion) << '.' << VK_VERSION_PATCH(p.vk10.apiVersion)
		    << "  driver=" << p.vk10.driverVersion << "  subgroup=" << p.vk11.subgroupSize << '\n';
		bool suitable = true;
		auto need = [&](bool present, const char *name) {
			if (!present) {
				log << "  missing: " << name << '\n';
				suitable = false;
			}
		};
		need(p.vk10.apiVersion >= VK_API_VERSION_1_3, "Vulkan 1.3");
		for (const char *name : extensions)
			need(gpu->GetExtensionSupport(name), name);
		if (reference) {
			log << "Emulator reference backend. Driver extensions:\n";
			for (const char *name : kExtensions)
				log << "  " << name << (gpu->GetExtensionSupport(name) ? " available" : " unavailable") << '\n';
		}
		std::optional<uint32_t> family;
		constexpr VkQueueFlags gfx_compute = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
		const auto &families = gpu->GetQueueFamilyProperties();
		for (uint32_t i = 0; i < families.size(); ++i) {
			VkBool32 present = false;
			if ((families[i].queueFlags & gfx_compute) == gfx_compute &&
			    vkGetPhysicalDeviceSurfaceSupportKHR(handle, i, surface, &present) == VK_SUCCESS && present) {
				family = i;
				break;
			}
		}
		need(family.has_value(), "graphics + compute + presentation queue");
		if (!reference) need(f.vk10.geometryShader, "geometryShader (fragment primitive ID)");
		need(f.vk10.fragmentStoresAndAtomics, "fragmentStoresAndAtomics");
		need(f.vk10.shaderStorageImageExtendedFormats, "shaderStorageImageExtendedFormats (RG32F)");
		if (!reference) need(f.vk10.shaderStorageBufferArrayDynamicIndexing, "shaderStorageBufferArrayDynamicIndexing");
		need(f.vk11.storageBuffer16BitAccess, "storageBuffer16BitAccess");
		need(f.vk12.imagelessFramebuffer && f.vk13.synchronization2, "imagelessFramebuffer and synchronization2");
		if (!reference) need(f.vk12.bufferDeviceAddress, "bufferDeviceAddress");
		if (!reference) need(f.vk12.hostQueryReset, "hostQueryReset");
		need(f.vk12.shaderFloat16 && f.vk12.vulkanMemoryModel, "shaderFloat16 and vulkanMemoryModel");
		if (!reference) {
			need(f.vk12.shaderSampledImageArrayNonUniformIndexing, "shaderSampledImageArrayNonUniformIndexing");
			need(f.vk12.shaderStorageBufferArrayNonUniformIndexing, "shaderStorageBufferArrayNonUniformIndexing");
		}
		need(f.vk13.maintenance4, "maintenance4");
		if (!reference) need(f.vk13.computeFullSubgroups, "computeFullSubgroups");
		constexpr VkSubgroupFeatureFlags ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT;
		need((p.vk11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) && (p.vk11.subgroupSupportedOperations & ops) == ops,
		     "compute subgroup basic, ballot and vote");
		need(p.vk10.limits.maxComputeWorkGroupInvocations >= 128 && p.vk10.limits.maxComputeWorkGroupSize[0] >= 128,
		     "128-lane compute workgroup");
		if (reference) need(p.vk10.limits.maxComputeSharedMemorySize >= 16384, "16 KiB workgroup memory");
		else need(p.vk13.maxSubgroupSize && 128 % p.vk13.maxSubgroupSize == 0, "full subgroups in 128 lanes");
		need(p.vk10.limits.maxPerStageDescriptorStorageBuffers >= 19 && p.vk10.limits.maxDescriptorSetStorageBuffers >= 19,
		     "19 storage buffers in the tracing pass");
		const auto format = [&](VkFormat fmt, VkFormatFeatureFlags flags, const char *name) {
			VkFormatProperties props{};
			vkGetPhysicalDeviceFormatProperties(handle, fmt, &props);
			need((props.optimalTilingFeatures & flags) == flags, name);
		};
		if (!reference) {
			format(VK_FORMAT_R32G32_UINT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT, "RG32UI visibility buffer");
			format(VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT, "D24S8 depth attachment");
		}
		format(VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT, "RGBA32F render target");
		format(VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, "RG32F storage image");
		VkNRCMatrixConfig tile{};
		VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
		if (!reference && gpu->GetExtensionSupport(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
			VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &matrix};
			vkGetPhysicalDeviceFeatures2(handle, &features);
			need(matrix.cooperativeMatrix, "cooperativeMatrix");
			VkPhysicalDeviceCooperativeMatrixPropertiesKHR stages{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
			VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &stages};
			vkGetPhysicalDeviceProperties2(handle, &props);
			need(stages.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT, "compute-stage cooperativeMatrix");
			tile = SelectMatrix(handle, &log);
			need(tile.m != 0, "usable FP16 subgroup matrix tile and training shared memory");
		}
		if (!suitable)
			continue;
		VkPhysicalDeviceRayQueryFeaturesKHR ray{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, &matrix};
		VkPhysicalDeviceAccelerationStructureFeaturesKHR accel{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &ray};
		if (!reference) {
			VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &accel};
			vkGetPhysicalDeviceFeatures2(handle, &features);
			need(ray.rayQuery && accel.accelerationStructure, "rayQuery and accelerationStructure");
			if (!suitable) continue;
			accel = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &ray, VK_TRUE};
			matrix.cooperativeMatrixRobustBufferAccess = VK_FALSE;
		}
		auto enabled = gpu->GetDefaultFeatures();
		enabled.vk11.storageBuffer16BitAccess = VK_TRUE;
		enabled.vk12.bufferDeviceAddress = !reference;
		enabled.vk12.shaderFloat16 = VK_TRUE;
		enabled.vk12.hostQueryReset = !reference;
		enabled.vk12.vulkanMemoryModel = VK_TRUE;
		enabled.vk12.shaderSampledImageArrayNonUniformIndexing = !reference;
		enabled.vk12.shaderStorageBufferArrayNonUniformIndexing = !reference;
		enabled.vk13.computeFullSubgroups = !reference;
		enabled.vk13.maintenance4 = VK_TRUE;
		enabled.vk13.pNext = reference ? nullptr : &accel;
		auto selector = [&](const auto &) { return std::vector<myvk::QueueSelection>{{&queue, *family, 0}}; };
		auto device = myvk::Device::Create(gpu, selector, enabled, extensions);
		if (device) {
			if (reference) log << "Selected emulator GPU: triangle intersection and scalar MLP.";
			else log << "Selected KHR matrix " << tile.m << 'x' << tile.n << 'x' << tile.k
			    << (tile.fp32_accumulator ? " FP32" : " FP16") << " accumulator.";
			report = log.str();
			return device;
		}
		log << "Logical device or allocator creation failed.\n";
	}
	report = log.str();
	if (report.empty()) report = "Android Vulkan driver reported no physical devices.";
	return {};
}
