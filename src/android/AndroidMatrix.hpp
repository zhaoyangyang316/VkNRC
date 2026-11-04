#pragma once

#include <volk.h>

struct VkNRCMatrixConfig {
	uint32_t m{}, n{}, k{};
	bool fp32_accumulator{};
};

VkNRCMatrixConfig GetAndroidMatrixConfig(VkPhysicalDevice device);
