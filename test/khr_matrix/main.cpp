#include "../../src/rg/KhrMatrixShape.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <vulkan/vulkan.h>
#define OK(x)                                                                                                \
    do {                                                                                                     \
        auto result = (x);                                                                                   \
        if (result != VK_SUCCESS) {                                                                          \
            std::cerr << #x << " " << result << "\n";                                                        \
            std::exit(2);                                                                                    \
        }                                                                                                    \
    } while (0)
struct MappedBuffer {
    VkBuffer handle;
    VkDeviceMemory memory;
    void *mapped;
};
std::vector<char> read(const char *path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
void save(const char *path, const void *v, size_t n) {
    std::ofstream output(path, std::ios::binary);
    output.write(static_cast<const char *>(v), n);
    output.close();
    if (!output) {
        std::cerr << "Cannot write " << path << '\n';
        std::exit(2);
    }
}
int main(int argc, char **argv) {
    if (argc < 3)
        return 1;
    auto spirv = read(argv[1]);
    const bool fp32 = std::strcmp(argv[2], "f32") == 0;
    const bool scalar = std::strcmp(argv[2], "scalar") == 0;
    if (spirv.empty())
        return 2;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    VkInstance instance;
    OK(vkCreateInstance(&instance_info, nullptr, &instance));
    uint32_t n = 0;
    OK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
    std::vector<VkPhysicalDevice> physical_devices(n);
    OK(vkEnumeratePhysicalDevices(instance, &n, physical_devices.data()));
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    auto get_matrix_properties = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
        instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!scalar && !get_matrix_properties)
        return 77;
    std::array<uint32_t, 3> tile_shape{};
    auto accumulator_type = fp32 ? VK_COMPONENT_TYPE_FLOAT32_KHR : VK_COMPONENT_TYPE_FLOAT16_KHR;
    for (const auto candidate : physical_devices) {
        if (scalar) {
            VkPhysicalDeviceVulkan12Features caps12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceVulkan11Features caps11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &caps12};
            VkPhysicalDeviceFeatures2 caps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &caps11};
            vkGetPhysicalDeviceFeatures2(candidate, &caps);
            if (!caps11.storageBuffer16BitAccess || !caps12.shaderFloat16 || !caps12.vulkanMemoryModel) continue;
            physical_device = candidate;
            break;
        }
        uint32_t count = 0;
        OK(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr));
        std::vector<VkExtensionProperties> extensions(count);
        OK(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, extensions.data()));
        bool supported = false;
        for (const auto &extension : extensions)
            supported |= std::strcmp(extension.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0;
        if (!supported)
            continue;
        OK(get_matrix_properties(candidate, &count, nullptr));
        std::vector<VkCooperativeMatrixPropertiesKHR> tiles(
            count, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        OK(get_matrix_properties(candidate, &count, tiles.data()));
        for (auto &t : tiles) {
            if (t.scope == VK_SCOPE_SUBGROUP_KHR && t.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                t.BType == t.AType && t.CType == accumulator_type && t.ResultType == accumulator_type &&
                !t.saturatingAccumulation && SupportsKhrMatrixShape(t.MSize, t.NSize, t.KSize)) {
                tile_shape = {t.MSize, t.NSize, t.KSize};
                if (t.MSize == 16 && t.NSize == 16 && t.KSize == 16)
                    break;
            }
        }
        if (tile_shape[0]) {
            physical_device = candidate;
            break;
        }
    }
    if (!physical_device)
        return 77;
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    std::cout << properties.deviceName << '\n';
    if (scalar) std::cout << "Scalar compute (cooperative matrix disabled)\n";
    else std::cout << "M,N,K " << tile_shape[0] << "," << tile_shape[1] << "," << tile_shape[2] << "\n";
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &n, families.data());
    uint32_t family = 0;
    while (family < families.size() && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT))
        ++family;
    if (family == families.size())
        return 77;
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR mat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    mat.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, scalar ? nullptr : &mat};
    f13.computeFullSubgroups = !scalar;
    f13.maintenance4 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.shaderFloat16 = f12.vulkanMemoryModel = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f11};
    const char *ext = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    device_info.enabledExtensionCount = scalar ? 0u : 1u;
    device_info.ppEnabledExtensionNames = scalar ? nullptr : &ext;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.queueCreateInfoCount = 1;
    VkDevice device;
    OK(vkCreateDevice(physical_device, &device_info, nullptr, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    const int sizes[] = {20672 * 2, 20672 * 4, 256 * 64 * 2, 256 * 16, 256 * 16};
    MappedBuffer buffers[5]{};
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (int i = 0; i < 5; ++i) {
        auto &b = buffers[i];
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = sizes[i];
        ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        OK(vkCreateBuffer(device, &ci, nullptr, &b.handle));
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, b.handle, &requirements);
        uint32_t memory_type = 0;
        auto mask = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        while (memory_type < memory_properties.memoryTypeCount &&
               (!(requirements.memoryTypeBits & (1u << memory_type)) ||
                ((memory_properties.memoryTypes[memory_type].propertyFlags & mask) != mask)))
            ++memory_type;
        if (memory_type == memory_properties.memoryTypeCount)
            return 5;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.memoryTypeIndex = memory_type;
        allocation.allocationSize = requirements.size;
        OK(vkAllocateMemory(device, &allocation, nullptr, &b.memory));
        OK(vkBindBufferMemory(device, b.handle, b.memory, 0));
        OK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
        std::memset(b.mapped, 0, sizes[i]);
    }
    for (auto [i, file] :
         std::vector<std::pair<int, const char *>>{{0, "weights.bin"}, {2, "input.bin"}, {3, "target.bin"}}) {
        auto bytes = read(file);
        if (bytes.size() != sizes[i])
            return 6;
        std::memcpy(buffers[i].mapped, bytes.data(), bytes.size());
    }
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (int i = 0; i < 5; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_layout_info.bindingCount = 5;
    set_layout_info.pBindings = bindings;
    VkDescriptorSetLayout set_layout;
    OK(vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout));
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.poolSizeCount = 1;
    VkDescriptorPool descriptor_pool;
    OK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool));
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &set_layout;
    VkDescriptorSet set;
    OK(vkAllocateDescriptorSets(device, &set_info, &set));
    for (int i = 0; i < 5; i++) {
        VkDescriptorBufferInfo buffer_info{buffers[i].handle, 0, (VkDeviceSize)sizes[i]};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = i;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo = &buffer_info;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    VkPipelineLayout layout;
    OK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &layout));
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = spirv.size();
    module_info.pCode = (const uint32_t *)spirv.data();
    VkShaderModule shader;
    OK(vkCreateShaderModule(device, &module_info, nullptr, &shader));
    VkSpecializationMapEntry tile_constants[3] = {{1, 0, 4}, {2, 4, 4}, {3, 8, 4}};
    VkSpecializationInfo specialization{3, tile_constants, sizeof(tile_shape), tile_shape.data()};
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.layout = layout;
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.flags = scalar ? 0u : VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader;
    pipeline_info.stage.pName = "main";
    pipeline_info.stage.pSpecializationInfo = scalar ? nullptr : &specialization;
    VkPipeline pipeline;
    OK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline));
    VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.queueFamilyIndex = family;
    VkCommandPool pool;
    OK(vkCreateCommandPool(device, &command_pool_info, nullptr, &pool));
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.commandBufferCount = 1;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VkCommandBuffer command;
    OK(vkAllocateCommandBuffers(device, &command_info, &command));
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    OK(vkBeginCommandBuffer(command, &begin_info));
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(command, 2, 1, 1);
    VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &host_barrier, 0, nullptr, 0, nullptr);
    OK(vkEndCommandBuffer(command));
    VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &command;
    OK(vkQueueSubmit(queue, 1, &submission, VK_NULL_HANDLE));
    OK(vkQueueWaitIdle(queue));
    save("prediction.bin", buffers[4].mapped, sizes[4]);
    save("gradient.bin", buffers[1].mapped, sizes[1]);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    for (auto &b : buffers) {
        vkUnmapMemory(device, b.memory);
        vkDestroyBuffer(device, b.handle, nullptr);
        vkFreeMemory(device, b.memory, nullptr);
    }
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}
