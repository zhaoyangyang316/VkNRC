#pragma once

#include <myvk/Buffer.hpp>
#include <algorithm>
#include <stdexcept>

// The device-address alignment for an AS build can exceed the storage buffer's
// memory binding alignment. Keep the aligned address inside our allocation.
struct VkBuildScratch {
	myvk::Ptr<myvk::Buffer> buffer;
	VkDeviceAddress address;

	static VkBuildScratch Create(const myvk::Ptr<myvk::Device> &device, VkDeviceSize size) {
		VkPhysicalDeviceAccelerationStructurePropertiesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
		VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &as};
		vkGetPhysicalDeviceProperties2(device->GetPhysicalDevicePtr()->GetHandle(), &props);
		VkDeviceSize mask = as.minAccelerationStructureScratchOffsetAlignment - 1;
		auto buffer = myvk::Buffer::Create(device, std::max<VkDeviceSize>(size, 1) + mask, 0,
		    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		if (!buffer) throw std::runtime_error("Could not allocate acceleration structure build scratch");
		return {buffer, (buffer->GetDeviceAddress() + mask) & ~mask};
	}
};
