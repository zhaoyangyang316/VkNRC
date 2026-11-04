#include "AndroidDevice.hpp"
#include "AndroidControls.hpp"
#include "AndroidFrameTools.hpp"
#include "../rg/NRCRenderGraph.hpp"
#include "../rg/ScreenPass.hpp"

#include <android/asset_manager.h>
#include <android/input.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <myvk/Instance.hpp>
#include <myvk_rg/resource/InputImage.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace ui = android_controls;
ui::Mailbox controls;
constexpr char kLog[] = "VkNRC";
#if defined(__x86_64__)
constexpr bool kReference = true;
#else
constexpr bool kReference = false;
#endif
constexpr std::array<VkExtent2D, 5> kPresets{{{160, 90}, {320, 180}, {640, 360}, {960, 540}, {1280, 720}}};
constexpr int kDefaultResolution = kReference ? 0 : 1;

void Check(VkResult result, const char *operation) {
	if (result != VK_SUCCESS)
		throw std::runtime_error(std::string(operation) + " failed (VkResult " + std::to_string(result) + ")");
}

void ShowError(ANativeActivity *activity, const std::string &message) {
	__android_log_print(ANDROID_LOG_ERROR, kLog, "%s", message.c_str());
	// NativeActivity callbacks run on the glue thread. The Activity dispatches the dialog to its UI thread.
	JNIEnv *env = nullptr;
	const bool attach = activity->vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_EDETACHED;
	if (attach && activity->vm->AttachCurrentThread(&env, nullptr) != JNI_OK)
		return;
	if (!env)
		return;
	jclass klass = env->GetObjectClass(activity->clazz);
	jmethodID show = klass ? env->GetMethodID(klass, "showRendererError", "(Ljava/lang/String;)V") : nullptr;
	jstring msg = show ? env->NewStringUTF(message.c_str()) : nullptr;
	if (msg)
		env->CallVoidMethod(activity->clazz, show, msg);
	if (env->ExceptionCheck()) {
		env->ExceptionDescribe();
		env->ExceptionClear();
	}
	if (msg) env->DeleteLocalRef(msg);
	if (klass) env->DeleteLocalRef(klass);
	if (attach) activity->vm->DetachCurrentThread();
}

Scene LoadCornell(ANativeActivity *activity) {
	// tinyobjloader resolves mtllib against the OBJ's filesystem directory.
	const auto directory = std::filesystem::path{activity->internalDataPath} / "cornell-box";
	std::filesystem::create_directories(directory);
	for (const char *file : {"cornell-box.obj", "cornell-box.mtl"}) {
		const auto name = std::string("cornell-box/") + file;
		std::unique_ptr<AAsset, decltype(&AAsset_close)> asset(
		    AAssetManager_open(activity->assetManager, name.c_str(), AASSET_MODE_STREAMING), AAsset_close);
		if (!asset)
			throw std::runtime_error("Missing packaged scene asset: " + name);
		std::ofstream output(directory / file, std::ios::binary | std::ios::trunc);
		std::array<char, 4096> buffer;
		int n;
		while ((n = AAsset_read(asset.get(), buffer.data(), buffer.size())) > 0)
			output.write(buffer.data(), n);
		output.close();
		if (n < 0 || !output)
			throw std::runtime_error("Could not extract scene asset: " + name);
	}
	auto scene = Scene::LoadOBJSingleInstance(directory / "cornell-box.obj");
	if (scene.Empty())
		throw std::runtime_error("Could not load the bundled Cornell Box");
	return scene;
}

// VkSwapchainKHR owns these images. MyVK owns only the views and framebuffer references.
class WindowImage final : public myvk::ImageBase {
	myvk::Ptr<myvk::Device> device_;
public:
	WindowImage(const myvk::Ptr<myvk::Device> &device, VkImage image, VkFormat format, VkExtent2D extent) : device_(device) {
		m_image = image;
		m_extent = {extent.width, extent.height, 1};
		m_format = format;
		m_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		m_type = VK_IMAGE_TYPE_2D;
		m_mip_levels = m_array_layers = 1;
	}
	const myvk::Ptr<myvk::Device> &GetDevicePtr() const override { return device_; }
};

class AndroidPresentGraph final : public myvk_rg::RenderGraphBase {
public:
	AndroidPresentGraph(const myvk::Ptr<myvk::Device> &device, const myvk::Ptr<myvk::ImageView> &frame,
	                    const myvk::Ptr<myvk::ImageView> &target) : RenderGraphBase(device) {
		auto source = CreateResource<myvk_rg::InputImage>({"linear"}, frame);
		source->SetSyncType(myvk_rg::ExternalSyncType::kCustom);
		source->SetSrcLayout(VK_IMAGE_LAYOUT_GENERAL);
		source->SetDstLayout(VK_IMAGE_LAYOUT_GENERAL);
		source->SetSrcPipelineStages(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
		source->SetSrcAccessFlags(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		source->SetDstPipelineStages(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
		source->SetDstAccessFlags(VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		auto output = CreateResource<myvk_rg::InputImage>({"window"}, target);
		output->SetSyncType(myvk_rg::ExternalSyncType::kCustom);
		output->SetSrcLayout(VK_IMAGE_LAYOUT_UNDEFINED);
		output->SetDstLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
		output->SetSrcPipelineStages(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
		output->SetDstPipelineStages(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
		auto present = CreatePass<rg::PresentScreenPass>(
		    {"present"}, rg::PresentScreenPass::Args{.linear_image = source->Alias(), .screen_image = output->Alias()});
		AddResult({"screen"}, present->GetScreenOutput());
	}
};

class AndroidRenderer {
	myvk::Ptr<myvk::Instance> instance_;
	VkSurfaceKHR surface_{VK_NULL_HANDLE};
	myvk::Ptr<myvk::Device> device_;
	myvk::Ptr<myvk::Queue> queue_;
	myvk::Ptr<VkScene> scene_;
	myvk::Ptr<VkSceneTLAS> tlas_;
	myvk::Ptr<Camera> camera_;
	myvk::Ptr<VkNRCState> nrc_;
	myvk::Ptr<rg::NRCRenderGraph> trace_;
	myvk::Ptr<myvk::CommandBuffer> command_;
	myvk::Ptr<myvk::Fence> completed_;
	myvk::Ptr<myvk::Semaphore> acquired_;
	VkSwapchainKHR swapchain_{VK_NULL_HANDLE};
	std::vector<myvk::Ptr<AndroidPresentGraph>> present_;
	std::vector<myvk::Ptr<myvk::ImageView>> targets_;
	VkExtent2D surface_extent_{};
	// Per swapchain image: reacquisition guarantees the preceding present wait has completed.
	std::vector<myvk::Ptr<myvk::Semaphore>> rendered_;
	bool first_frame_{true};
	uint32_t frame_count_{}, displayed_spp_{};
	int resolution_{kDefaultResolution};
	bool paused_{}, locked_{}, train_once_{}, has_frame_{}, redraw_{true};
	std::filesystem::path captures_;
	AndroidFrameTools frame_tools_;
	bool diagnostic_due_{};

	void ReportTraining() {
		struct Readback {
			VmaAllocator allocator;
			VkBuffer buffer{};
			VmaAllocation allocation{};
			~Readback() { if (buffer) vmaDestroyBuffer(allocator, buffer, allocation); }
		} dst{device_->GetAllocatorHandle()};
		VkBufferCreateInfo buffer{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(uint32_t),
		                          .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
		VmaAllocationCreateInfo alloc{.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
		                              .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
		VmaAllocationInfo info{};
		Check(vmaCreateBuffer(dst.allocator, &buffer, &alloc, &dst.buffer, &dst.allocation, &info), "Allocate training readback");
		if (!info.pMappedData) throw std::runtime_error("Training diagnostic buffer is not host-mapped");
		Check(command_->Reset(), "Reset training readback command");
		Check(command_->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT), "Begin training readback command");
		const VkMemoryBarrier before{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		                             .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
		vkCmdPipelineBarrier(command_->GetHandle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     0, 1, &before, 0, nullptr, 0, nullptr);
		const VkBufferCopy copy{.size = sizeof(uint32_t)};
		vkCmdCopyBuffer(command_->GetHandle(), nrc_->GetOptimizerStateBuffer()->GetHandle(), dst.buffer, 1, &copy);
		const VkMemoryBarrier after{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		                            .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
		vkCmdPipelineBarrier(command_->GetHandle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
		                     0, 1, &after, 0, nullptr, 0, nullptr);
		Check(command_->End(), "End training readback command");
		Check(completed_->Reset(), "Reset training readback fence");
		Check(command_->Submit(completed_), "Submit training readback");
		Check(completed_->Wait(), "Complete training readback");
		Check(vmaInvalidateAllocation(dst.allocator, dst.allocation, 0, VK_WHOLE_SIZE), "Invalidate training readback");
		__android_log_print(ANDROID_LOG_INFO, kLog, "NRC training: frame=%u optimizer_step=%u", frame_count_,
		                    *static_cast<const uint32_t *>(info.pMappedData));
	}

	void DropSwapchain() {
		present_.clear();
		targets_.clear();
		rendered_.clear();
		if (swapchain_)
			vkDestroySwapchainKHR(device_->GetHandle(), swapchain_, nullptr);
		swapchain_ = VK_NULL_HANDLE;
	}
	void MakePresentGraphs() {
		present_.clear();
		for (const auto &view : targets_) {
			auto graph = myvk::MakePtr<AndroidPresentGraph>(device_, nrc_->GetAccumulateImageView(), view);
			graph->SetCanvasSize(surface_extent_);
			present_.push_back(std::move(graph));
		}
	}
	void MakeTraceGraph() {
		trace_ = myvk::MakePtr<rg::NRCRenderGraph>(scene_, nrc_, camera_, tlas_);
		trace_->SetCanvasSize(kPresets[resolution_]);
	}
	void Status(const std::string &message) {
		controls.Status(message);
		__android_log_print(ANDROID_LOG_INFO, kLog, "%s", message.c_str());
	}
	void Publish() const {
		controls.Publish({nrc_->GetLeftMethod(), nrc_->GetRightMethod(), nrc_->IsUseNEE(), nrc_->IsFastMode(),
		                  int(std::lround(1000 * nrc_->GetMinRoughness())), nrc_->IsIndirectOnly(), int(nrc_->GetMaxBounces()),
		                  resolution_, nrc_->IsAccumulate(), int(nrc_->GetAccumulationWindow()), paused_, locked_,
	                  nrc_->IsUseEMAWeights(), has_frame_ ? int(displayed_spp_) : 0,
		                  int(nrc_->GetMaxAccumulationWindow()), swapchain_ != VK_NULL_HANDLE, frame_tools_.IsBenchmarkActive()});
	}
	void ResizeRender(int preset) {
		if (preset < 0 || preset >= int(kPresets.size()) || preset == resolution_) return;
		const auto extent = kPresets[preset];
		const auto &limits = device_->GetPhysicalDevicePtr()->GetProperties().vk10.limits;
		if (VkNRCState::GetEvalRecordBufferSize(extent) > limits.maxStorageBufferRange ||
		    std::max(extent.width, extent.height) > limits.maxImageDimension2D) {
			Status("This render resolution exceeds the GPU's limits");
			return;
		}
		try {
			nrc_->ResetAccumulateImage(extent);
		} catch (const std::runtime_error &error) {
			Status(error.what()); // ResetAccumulateImage commits only after both images are usable.
			return;
		}
		resolution_ = preset;
		has_frame_ = false;
		frame_tools_.ResetFrame();
		MakeTraceGraph();
		MakePresentGraphs();
		Status("Render resolution: " + std::to_string(extent.width) + "x" + std::to_string(extent.height));
	}
	void Apply(ui::Command command) {
		const auto [id, v] = command;
		if (id == ui::Capture) {
			Status(frame_tools_.Capture(queue_, captures_, v != 0));
			return;
		}
		if (id == ui::StopBenchmark) { CancelBenchmark(); return; }
		if (id == ui::Benchmark) {
			if (paused_) Status("Resume rendering to start a benchmark");
			else if (!frame_tools_.IsBenchmarkActive()) Status(frame_tools_.StartBenchmark(std::clamp(v, 1, 120)));
			return;
		}
		if (id == ui::Pause) {
			CancelBenchmark();
			paused_ = v != 0;
			Status(paused_ ? "Rendering paused" : "Rendering resumed");
			return;
		}
		if (paused_) { Status("Resume rendering to change settings"); return; }
		CancelBenchmark();
		bool restart_training = false;
		switch (id) {
		case ui::Left: case ui::Right: {
			auto method = static_cast<VkNRCState::Method>(std::clamp(v, 0, 2));
			if ((id == ui::Left ? nrc_->GetLeftMethod() : nrc_->GetRightMethod()) == method) return;
			if (id == ui::Left) nrc_->SetLeftMethod(method); else nrc_->SetRightMethod(method);
			break;
		}
		case ui::CycleView: {
			const auto l = nrc_->GetLeftMethod(), r = nrc_->GetRightMethod();
			const bool split = l == VkNRCState::kNRC && r == VkNRCState::kNone;
			const bool pt = l == VkNRCState::kNone && r == VkNRCState::kNone;
			nrc_->SetLeftMethod(split ? VkNRCState::kNone : VkNRCState::kNRC);
			nrc_->SetRightMethod(pt ? VkNRCState::kNRC : VkNRCState::kNone);
			Status(split ? "View: full PT" : pt ? "View: full NRC" : "View: split NRC/PT");
			break;
		}
		case ui::Nee: restart_training = nrc_->SetUseNEE(v != 0); if (!restart_training) return; break;
		case ui::Fast: case ui::Roughness:
			restart_training = nrc_->SetFastMode(id == ui::Fast ? v != 0 : nrc_->IsFastMode(),
			    id == ui::Roughness ? std::clamp(v, 0, 1000) / 1000.f : nrc_->GetMinRoughness());
			if (!restart_training) return;
			break;
		case ui::Bounces: restart_training = nrc_->SetMaxBounces(std::clamp(v, 1, 64)); if (!restart_training) return; break;
		case ui::Indirect: nrc_->SetIndirectOnly(v != 0); return;
		case ui::Resolution: ResizeRender(v); return;
		case ui::Accumulate: if (nrc_->IsAccumulate() == (v != 0)) return; nrc_->SetAccumulate(v != 0); break;
		case ui::Window: {
			const uint32_t window = std::clamp(v, 0, int(nrc_->GetMaxAccumulationWindow()));
			if (window != nrc_->GetAccumulationWindow() && !nrc_->SetAccumulationWindow(window))
				Status("Could not allocate the selected accumulation window");
			else if (window) nrc_->SetAccumulate(true);
			return;
		}
		case ui::Lock: if (locked_ == (v != 0)) return; locked_ = v != 0; train_once_ = false; break;
		case ui::Ema: if (nrc_->IsUseEMAWeights() == (v != 0)) return; nrc_->SetUseEMAWeights(v != 0); break;
		case ui::TrainOneFrame: if (!locked_) return; train_once_ = true; Status("Training one frame"); break;
		case ui::Restart: nrc_->SetAccumulate(true); Status("Restarted accumulation"); break;
		case ui::Retrain: nrc_->ResetMLPBuffers(); Status("Reset NRC weights"); break;
		default: return;
		}
		if (restart_training) {
			nrc_->ResetMLPBuffers();
			locked_ = train_once_ = false;
			Status("Restarted cache training for the new settings");
		}
		nrc_->ResetAccumulateCount();
	}
	void CreateSurface(ANativeWindow *window) {
		VkAndroidSurfaceCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR, .window = window};
		Check(vkCreateAndroidSurfaceKHR(instance_->GetHandle(), &info, nullptr, &surface_), "Create Android surface");
	}

public:
	~AndroidRenderer() {
		if (device_) device_->WaitIdle();
		DropSwapchain();
		if (surface_) vkDestroySurfaceKHR(instance_->GetHandle(), surface_, nullptr);
	}

	void Initialize(android_app *app) {
		captures_ = std::filesystem::path(app->activity->externalDataPath ? app->activity->externalDataPath :
		                                  app->activity->internalDataPath) / "captures";
		// Extract before probing Vulkan so the Activity can show a CPU preview on unsupported devices.
		auto cornell = LoadCornell(app->activity);
		Check(volkInitialize(), "Load Android Vulkan driver");
		uint32_t version = VK_API_VERSION_1_0;
		if (vkEnumerateInstanceVersion)
			Check(vkEnumerateInstanceVersion(&version), "Query Vulkan loader version");
		if (version < VK_API_VERSION_1_3)
			throw std::runtime_error("VkNRC needs Vulkan 1.3. The Android loader reports " +
			                         std::to_string(VK_VERSION_MAJOR(version)) + "." + std::to_string(VK_VERSION_MINOR(version)) + ".");
		instance_ = myvk::Instance::Create({VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME});
		if (!instance_)
			throw std::runtime_error("Could not create a Vulkan 1.3 Android surface-capable instance");
		CreateSurface(app->window);
		std::string report;
		device_ = CreateAndroidDevice(instance_, surface_, queue_, report, kReference);
		if (!device_)
			throw std::runtime_error(report);
		__android_log_print(ANDROID_LOG_INFO, kLog, "%s", report.c_str());
		scene_ = myvk::MakePtr<VkScene>(queue_, cornell);
		if (!kReference)
			tlas_ = myvk::MakePtr<VkSceneTLAS>(myvk::MakePtr<VkSceneBLAS>(scene_));
		nrc_ = myvk::MakePtr<VkNRCState>(queue_, kPresets[resolution_], VkDeviceSize{64} << 20);
		nrc_->SetRandomSeed(1337);
		nrc_->ResetMLPBuffers();
		nrc_->SetAccumulate(true);
		nrc_->SetLeftMethod(VkNRCState::kNRC);
		nrc_->SetRightMethod(VkNRCState::kNone);
		camera_ = myvk::MakePtr<Camera>();
		camera_->position = {0.f, -0.005007153f, 3.86123033f};
		camera_->fov = glm::radians(39.30765f);
		MakeTraceGraph();
		auto pool = myvk::CommandPool::Create(queue_, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		if (!pool) throw std::runtime_error("Could not allocate the render command pool");
		command_ = myvk::CommandBuffer::Create(pool);
		completed_ = myvk::Fence::Create(device_, VK_FENCE_CREATE_SIGNALED_BIT);
		acquired_ = myvk::Semaphore::Create(device_);
		if (!command_ || !completed_ || !acquired_) throw std::runtime_error("Could not allocate Vulkan frame resources");
	}

	bool MakeSwapchain(ANativeWindow *window) {
		CancelBenchmark();
		if (!surface_) CreateSurface(window);
		const auto physical = device_->GetPhysicalDevicePtr()->GetHandle();
		VkSurfaceCapabilitiesKHR caps{};
		Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface_, &caps), "Query surface capabilities");
		VkExtent2D extent = caps.currentExtent;
		if (extent.width == UINT32_MAX) {
			const int w = ANativeWindow_getWidth(window), h = ANativeWindow_getHeight(window);
			if (w <= 0 || h <= 0) return false;
			extent = {std::clamp(uint32_t(w), caps.minImageExtent.width, caps.maxImageExtent.width),
			          std::clamp(uint32_t(h), caps.minImageExtent.height, caps.maxImageExtent.height)};
		}
		if (!extent.width || !extent.height) return false;
		Check(device_->WaitIdle(), "Wait before resizing surface");
		DropSwapchain();
		surface_extent_ = extent;
		uint32_t count = 0;
		Check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &count, nullptr), "Query surface formats");
		std::vector<VkSurfaceFormatKHR> formats(count);
		if (!count) throw std::runtime_error("Android surface has no Vulkan formats");
		Check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &count, formats.data()), "Get surface formats");
		formats.resize(count);
		auto format = std::find_if(formats.begin(), formats.end(), [](auto f) {
			return f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
			       (f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_UNDEFINED);
		});
		if (format == formats.end()) throw std::runtime_error("Android surface lacks an RGBA8/BGRA8 UNORM presentation format");
		if (format->format == VK_FORMAT_UNDEFINED) format->format = VK_FORMAT_R8G8B8A8_UNORM;
		uint32_t image_count = caps.minImageCount + 1;
		if (caps.maxImageCount) image_count = std::min(image_count, caps.maxImageCount);
		const auto alpha = static_cast<VkCompositeAlphaFlagBitsKHR>(caps.supportedCompositeAlpha & -caps.supportedCompositeAlpha);
		VkSwapchainCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = surface_,
		                              .minImageCount = image_count, .imageFormat = format->format,
		                              .imageColorSpace = format->colorSpace, .imageExtent = extent, .imageArrayLayers = 1,
		                              .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		                              .preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
		                                                  ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform,
		                              .compositeAlpha = alpha, .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = VK_TRUE};
		Check(vkCreateSwapchainKHR(device_->GetHandle(), &info, nullptr, &swapchain_), "Create Android swapchain");
		Check(vkGetSwapchainImagesKHR(device_->GetHandle(), swapchain_, &count, nullptr), "Query swapchain images");
		std::vector<VkImage> images(count);
		Check(vkGetSwapchainImagesKHR(device_->GetHandle(), swapchain_, &count, images.data()), "Get swapchain images");
		for (VkImage handle : images) {
			auto image = myvk::MakePtr<WindowImage>(device_, handle, format->format, extent);
			auto view = myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D);
			auto signal = myvk::Semaphore::Create(device_);
			if (!view || !signal) throw std::runtime_error("Could not create swapchain image resources");
			targets_.push_back(std::move(view));
			rendered_.push_back(std::move(signal));
		}
		MakePresentGraphs();
		redraw_ = true;
		Publish();
		__android_log_print(ANDROID_LOG_INFO, kLog, "Cornell Box: %s backend, render %ux%u, surface %ux%u (tap to cycle view)",
		                    kReference ? "scalar NRC + compute traversal" : "KHR matrix + ray query",
		                    kPresets[resolution_].width, kPresets[resolution_].height, extent.width, extent.height);
		return true;
	}
	bool Sleeping() const { return paused_ && has_frame_ && !redraw_; }
	void RequestRedraw() { redraw_ = true; }
	void CancelBenchmark() {
		if (frame_tools_.IsBenchmarkActive()) {
			frame_tools_.CancelBenchmark();
			Status("Benchmark canceled");
			Publish();
		}
	}
	void DetachWindow() {
		CancelBenchmark();
		device_->WaitIdle();
		DropSwapchain();
		if (surface_) vkDestroySurfaceKHR(instance_->GetHandle(), surface_, nullptr);
		surface_ = VK_NULL_HANDLE;
		Publish(); // Keep the scene, weights, render image, history and controls across window re-creation.
	}

	// Single in-flight frame: the training counters are host-mapped and reused after the fence.
	bool Draw() {
		VkResult wait = completed_->Wait(16'000'000);
		if (wait == VK_TIMEOUT) return true;
		Check(wait, "Wait for rendered frame");
		auto commands = controls.Take();
		for (auto command : commands) Apply(command);
		if (!commands.empty()) {
			Publish();
			__android_log_print(ANDROID_LOG_INFO, kLog,
			    "Settings: view=%d/%d nee=%d fast=%d min=%.3f indirect=%d depth=%u render=%ux%u accumulate=%d window=%u paused=%d lock=%d ema=%d",
			    nrc_->GetLeftMethod(), nrc_->GetRightMethod(), nrc_->IsUseNEE(), nrc_->IsFastMode(), nrc_->GetMinRoughness(),
			    nrc_->IsIndirectOnly(), nrc_->GetMaxBounces(), kPresets[resolution_].width, kPresets[resolution_].height,
			    nrc_->IsAccumulate(), nrc_->GetAccumulationWindow(), paused_, locked_, nrc_->IsUseEMAWeights());
		}
		if (Sleeping()) return true;
		if (diagnostic_due_ && !frame_tools_.IsBenchmarkActive()) { ReportTraining(); diagnostic_due_ = false; }
		uint32_t index;
		auto result = vkAcquireNextImageKHR(device_->GetHandle(), swapchain_, 16'000'000, acquired_->GetHandle(), VK_NULL_HANDLE, &index);
		if (result == VK_TIMEOUT || result == VK_NOT_READY) return true;
		if (result == VK_ERROR_OUT_OF_DATE_KHR) return false;
		if (result != VK_SUBOPTIMAL_KHR) Check(result, "Acquire Android image");
		Check(command_->Reset(), "Reset frame command");
		Check(command_->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT), "Begin frame command");
		const bool render = !paused_ || !has_frame_;
		if (render) {
			nrc_->SetTrainProbability(!locked_ || train_once_ ? VkNRCState::GetDefaultTrainProbability() : 0.f);
			trace_->CmdExecute(command_);
		}
		present_[index]->CmdExecute(command_);
		Check(command_->End(), "End frame command");
		Check(completed_->Reset(), "Reset frame fence");
		Check(command_->Submit({{acquired_, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {rendered_[index]}, completed_), "Submit frame");
		VkSemaphore signal = rendered_[index]->GetHandle();
		VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
		                      .pWaitSemaphores = &signal, .swapchainCount = 1, .pSwapchains = &swapchain_, .pImageIndices = &index};
		result = vkQueuePresentKHR(queue_->GetHandle(), &info);
		if (render) {
			nrc_->NextFrame();
			train_once_ = false;
			++frame_count_;
			diagnostic_due_ = frame_count_ == 4 || frame_count_ == 128;
		}
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			// The trace may have overwritten the capture image without successfully presenting its new settings.
			CancelBenchmark();
			if (render) { frame_tools_.ResetFrame(); has_frame_ = false; }
			return false;
		}
		if (result != VK_SUBOPTIMAL_KHR) Check(result, "Present Android image");
		redraw_ = false;
		if (render) {
			has_frame_ = true;
			displayed_spp_ = nrc_->GetRenderedSPP();
			if (auto result = frame_tools_.FinishRenderedFrame(queue_, *nrc_, "Cornell Box")) Status(*result);
		}
		Publish();
		if (first_frame_) {
			controls.Status("Renderer ready");
			__android_log_print(ANDROID_LOG_INFO, kLog, "First Cornell NRC frame presented (split NRC/PT)");
			first_frame_ = false;
		}
		return true;
	}
};

struct ActivityState {
	std::unique_ptr<AndroidRenderer> renderer;
	bool resumed{}, resize{true}, failed{};
};

void HandleCommand(android_app *app, int32_t event) {
	auto &state = *static_cast<ActivityState *>(app->userData);
	switch (event) {
	case APP_CMD_RESUME: state.resumed = true; break;
	case APP_CMD_PAUSE: state.resumed = false; if (state.renderer) state.renderer->CancelBenchmark(); break;
	case APP_CMD_INIT_WINDOW:
	case APP_CMD_WINDOW_RESIZED: state.resize = true; break;
	case APP_CMD_WINDOW_REDRAW_NEEDED: if (state.renderer) state.renderer->RequestRedraw(); break;
	case APP_CMD_TERM_WINDOW: if (state.renderer) state.renderer->DetachWindow(); state.resize = true; break;
	}
}
} // namespace

extern "C" JNIEXPORT void JNICALL Java_org_vknrc_demo_VkNRCActivity_setRendererOption(JNIEnv *, jclass, jint id, jint value) {
	controls.Enqueue({id, value});
}
extern "C" JNIEXPORT jintArray JNICALL Java_org_vknrc_demo_VkNRCActivity_getRendererOptions(JNIEnv *env, jclass) {
	const auto state = controls.Get();
	jintArray result = env->NewIntArray(state.size());
	if (result) env->SetIntArrayRegion(result, 0, state.size(), state.data());
	return result;
}
extern "C" JNIEXPORT jstring JNICALL Java_org_vknrc_demo_VkNRCActivity_getRendererStatus(JNIEnv *env, jclass) {
	return env->NewStringUTF(controls.Status().c_str());
}

extern "C" void android_main(android_app *app) {
	app_dummy(); // Retain the NativeActivity glue entry point when statically linked.
	ActivityState state;
	controls.Bind(app->looper, kDefaultResolution);
	app->userData = &state;
	app->onAppCmd = HandleCommand;
	app->onInputEvent = [](android_app *a, AInputEvent *event) -> int32_t {
		auto &s = *static_cast<ActivityState *>(a->userData);
		if (s.renderer && AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
			if ((AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK) == AMOTION_EVENT_ACTION_UP)
				controls.Enqueue({ui::CycleView, 0});
			return 1;
		}
		return 0;
	};
	while (!app->destroyRequested) {
		const bool active = app->window && state.resumed && !state.failed;
		const bool sleeping = active && !state.resize && state.renderer && state.renderer->Sleeping() && !controls.Pending();
		android_poll_source *source = nullptr;
		int events = 0;
		ALooper_pollOnce(active && !sleeping ? (state.resize ? 16 : 0) : -1, nullptr, &events, reinterpret_cast<void **>(&source));
		if (source) source->process(app, source);
		if (!app->window || !state.resumed || state.failed || app->destroyRequested) continue;
		try {
			if (!state.renderer) {
				state.renderer = std::make_unique<AndroidRenderer>();
				state.renderer->Initialize(app);
			}
			if (state.resize) {
				if (!state.renderer->MakeSwapchain(app->window)) continue;
				state.resize = false;
			}
			state.resize = !state.renderer->Draw();
		} catch (const std::exception &error) {
			state.failed = true;
			state.renderer.reset();
			controls.Close();
			controls.Status(error.what());
			ShowError(app->activity, error.what());
		}
	}
	controls.Close();
}
