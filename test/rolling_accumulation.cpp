// Offscreen GPU regression test for the production ScreenPass and history state.
// Usage: VkNRCRollingTest [output-directory] [--validation]
// No window or neural rendering is used. CPU truth is the explicit last-N list.
#include "../src/RenderCapture.hpp"
#include "../src/VkNRCState.hpp"
#include "../src/rg/ScreenPass.hpp"

#include <myvk/Instance.hpp>
#include <myvk_rg/resource/InputImage.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using RGB = std::array<float, 3>;
std::atomic<uint32_t> g_validation_errors{};

VKAPI_ATTR VkBool32 VKAPI_CALL ValidationCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT,
                                                 const VkDebugUtilsMessengerCallbackDataEXT *message, void *) {
	if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
		++g_validation_errors;
		std::cerr << "Vulkan validation: " << message->pMessage << '\n';
	}
	return VK_FALSE;
}

void Require(bool condition, const std::string &message) {
	if (!condition)
		throw std::runtime_error(message);
}

void Check(VkResult result, const char *operation) {
	Require(result == VK_SUCCESS, std::string(operation) + " failed: " + std::to_string(result));
}

class TestGraph final : public myvk_rg::RenderGraphBase {
	myvk::Ptr<VkNRCState> m_state;

public:
	TestGraph(const myvk::Ptr<myvk::Device> &device, const myvk::Ptr<VkNRCState> &state,
	          const myvk::Ptr<myvk::ImageView> &input) : RenderGraphBase(device), m_state(state) {
		auto source = CreateResource<myvk_rg::InputImage>({"input"}, input);
		source->SetSyncType(myvk_rg::ExternalSyncType::kCustom);
		source->SetSrcLayout(VK_IMAGE_LAYOUT_GENERAL);
		source->SetDstLayout(VK_IMAGE_LAYOUT_GENERAL);
		source->SetSrcPipelineStages(VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		source->SetSrcAccessFlags(VK_ACCESS_2_TRANSFER_WRITE_BIT);
		source->SetDstPipelineStages(VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		source->SetDstAccessFlags(VK_ACCESS_2_TRANSFER_WRITE_BIT);
		auto accumulate = CreateResource<myvk_rg::InputImage>({"accumulate"}, state->GetAccumulateImageView());
		auto history = CreateResource<myvk_rg::InputImage>({"history"}, state->GetHistoryImageView());
		auto output = CreateResource<myvk_rg::ManagedImage>({"output"}, VK_FORMAT_R32G32B32A32_SFLOAT);
		auto pass = CreatePass<rg::ScreenPass>(
		    {"screen"}, rg::ScreenPass::Args{.nrc_state_ptr = m_state,
		                                    .accumulate_image = accumulate->Alias(),
		                                    .history_image = history->Alias(),
		                                    .color_image = source->Alias(),
		                                    .screen_image = output->Alias()});
		AddResult({"output"}, pass->GetScreenOutput());
	}
	void PreExecute() const final {
		GetResource<myvk_rg::InputImage>({"accumulate"})->SetVkImageView(m_state->GetAccumulateImageView());
		GetResource<myvk_rg::InputImage>({"history"})->SetVkImageView(m_state->GetHistoryImageView());
	}
};

class Harness {
	myvk::Ptr<myvk::Queue> m_queue;
	myvk::Ptr<myvk::ImageView> m_input;
	myvk::Ptr<TestGraph> m_graph;
	bool m_input_initialized = false;
	VkExtent2D m_extent;
	std::deque<RGB> m_reference;
	uint32_t m_frames = 0;
	std::filesystem::path m_capture;
	uint32_t m_asserted_frames = 0;

	void RebuildInput() {
		auto image = myvk::Image::CreateTexture2D(m_queue->GetDevicePtr(), m_extent, 1,
		                                       VK_FORMAT_R32G32B32A32_SFLOAT,
		                                       VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
		m_input = myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D);
		m_graph = myvk::MakePtr<TestGraph>(m_queue->GetDevicePtr(), state, m_input);
		m_graph->SetCanvasSize(m_extent);
		m_input_initialized = false;
	}

public:
	myvk::Ptr<VkNRCState> state;
	Harness(const myvk::Ptr<myvk::Queue> &queue, const std::filesystem::path &directory)
	    : m_queue(queue), m_extent{8, 8}, m_capture(directory / "actual.pfm"),
	      state(myvk::MakePtr<VkNRCState>(queue, m_extent)) {
		state->SetAccumulate(true);
		RebuildInput();
	}
	void ResetReference() {
		m_reference.clear();
		m_frames = 0;
	}
	void SetWindow(uint32_t window) {
		Check(m_queue->WaitIdle(), "window change wait");
		Require(state->SetAccumulationWindow(window), "Test expects a changed accumulation window");
		Require(state->GetAccumulationWindow() == window, "Unexpected window clamp at 8x8 test resolution");
		Require(state->GetAccumulateCount() == 0 && state->GetHistoryWriteIndex() == 0,
		        "Changing the window did not reset history state");
		ResetReference();
	}
	void Reset() {
		state->ResetAccumulateCount();
		ResetReference();
	}
	void Resize(VkExtent2D extent) {
		Check(m_queue->WaitIdle(), "resize wait");
		m_extent = extent;
		state->ResetAccumulateImage(extent);
		ResetReference();
		RebuildInput();
	}
	void CheckCapture(const std::string &label) {
		Require(!m_reference.empty(), "No reference sample");
		Require(CaptureAccumulationPFM(m_queue, state->GetAccumulateImageView(), m_capture), "Capture failed");
		std::ifstream input(m_capture, std::ios::binary);
		std::string magic;
		uint32_t width = 0, height = 0;
		float scale = 0;
		input >> magic >> width >> height >> scale;
		input.get();
		Require(magic == "PF" && width == m_extent.width && height == m_extent.height && scale == -1.0f,
		        "Unexpected PFM header");
		std::vector<float> actual(size_t(width) * height * 3);
		input.read(reinterpret_cast<char *>(actual.data()), std::streamsize(actual.size() * sizeof(float)));
		Require(bool(input), "Incomplete PFM capture");
		std::array<double, 3> expected{};
		for (const RGB &sample : m_reference)
			for (unsigned c = 0; c < 3; ++c)
				expected[c] += sample[c];
		for (double &value : expected)
			value /= double(m_reference.size());
		for (size_t i = 0; i < actual.size(); ++i) {
			const double error = std::abs(double(actual[i]) - expected[i % 3]);
			const double tolerance = 2e-6 + 2e-6 * std::abs(expected[i % 3]);
			if (!std::isfinite(actual[i]) || error > tolerance) {
				std::ostringstream message;
				message << label << ", frame " << m_frames << ", element " << i << ": expected "
				        << expected[i % 3] << ", got " << actual[i] << "; capture: " << m_capture.string();
				throw std::runtime_error(message.str());
			}
		}
	}
	void Render(const RGB &color, const std::string &label) {
		auto pool = myvk::CommandPool::Create(m_queue);
		auto command = myvk::CommandBuffer::Create(pool);
		Check(command->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT), "command begin");
		const auto &image = m_input->GetImagePtr();
		command->CmdPipelineBarrier(
		    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, {}, {},
		    {image->GetMemoryBarrier(m_input->GetSubresourceRange(),
		                             m_input_initialized ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : 0,
		                             VK_ACCESS_TRANSFER_WRITE_BIT,
		                             m_input_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
		                             VK_IMAGE_LAYOUT_GENERAL)});
		VkClearColorValue clear{};
		for (unsigned c = 0; c < 3; ++c)
			clear.float32[c] = color[c];
		clear.float32[3] = 1.0f;
		auto range = m_input->GetSubresourceRange();
		vkCmdClearColorImage(command->GetHandle(), image->GetHandle(), VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
		m_graph->CmdExecute(command);
		Check(command->End(), "command end");
		auto fence = myvk::Fence::Create(m_queue->GetDevicePtr());
		Check(command->Submit(fence), "submit");
		Check(fence->Wait(), "render wait");
		m_input_initialized = true;
		state->NextFrame();
		++m_frames;
		m_reference.push_back(color);
		const auto window = state->GetAccumulationWindow();
		if (!state->IsAccumulate()) {
			while (m_reference.size() > 1)
				m_reference.pop_front();
		} else if (window > 0) {
			while (m_reference.size() > window)
				m_reference.pop_front();
		}
		const uint32_t expected_count = !state->IsAccumulate() ? 0 : window ? std::min(m_frames, window) : m_frames;
		const uint32_t expected_index = state->IsAccumulate() && window ? m_frames % window : 0;
		Require(state->GetAccumulateCount() == expected_count, label + ": wrong displayed sample count");
		Require(state->GetHistoryWriteIndex() == expected_index, label + ": wrong ring write index");
		CheckCapture(label);
		++m_asserted_frames;
	}
	uint32_t AssertedFrames() const { return m_asserted_frames; }
};
} // namespace

int main(int argc, char **argv) {
	const std::filesystem::path directory = argc > 1 ? argv[1] : "rolling-validation";
	const bool validation = argc > 2 && std::string_view(argv[2]) == "--validation";
	std::filesystem::create_directories(directory);
	std::ofstream report(directory / "report.txt");
	try {
		spdlog::set_level(spdlog::level::warn);
		auto instance = myvk::Instance::Create({}, validation, ValidationCallback);
		Require(bool(instance), "Instance creation failed");
		auto devices = myvk::PhysicalDevice::Fetch(instance);
		Require(!devices.empty(), "No Vulkan device");
		myvk::Ptr<myvk::Queue> queue;
		auto device = myvk::Device::Create(devices.front(), myvk::GenericQueueSelector{&queue},
		                                   devices.front()->GetDefaultFeatures(), {});
		Require(bool(device) && bool(queue), "Device creation failed");
		std::cout << "GPU: " << devices.front()->GetProperties().vk10.deviceName << '\n';
		Harness test(queue, directory);
		const auto pass = [&](const char *name) { std::cout << "PASS " << name << '\n'; report << "PASS " << name << '\n'; };
		for (uint32_t window : {1u, 3u, 4u, 64u}) {
			test.SetWindow(window);
			for (uint32_t frame = 0; frame < 3 * window + 5; ++frame) {
				RGB color = frame == 0 ? RGB{4096.0f, 256.0f, 16.0f}
				                       : RGB{float(frame % 7) * 0.25f, float(frame % 5) * 0.5f, float(frame % 3) * 0.125f};
				test.Render(color, "window " + std::to_string(window));
			}
			pass(("partial fill, spike expiry and three wraps N=" + std::to_string(window)).c_str());
		}
		test.SetWindow(4);
		for (unsigned i = 0; i < 7; ++i)
			test.Render({128, 64, 32}, "pre-reset");
		test.Reset();
		test.Render({0.125f, 0.25f, 0.5f}, "reset discards stale history");
		pass("reset discards stale ring samples");
		const auto count = test.state->GetAccumulateCount(), index = test.state->GetHistoryWriteIndex(), seed = test.state->GetSeed();
		for (unsigned i = 0; i < 100; ++i)
			test.state->NextFrame(false);
		Require(count == test.state->GetAccumulateCount() && index == test.state->GetHistoryWriteIndex() &&
		            seed == test.state->GetSeed(), "Pause advanced state");
		test.CheckCapture("pause leaves image unchanged");
		pass("pause freezes count, ring index, RNG and image");
		test.Resize({13, 5});
		for (unsigned i = 0; i < 11; ++i)
			test.Render({float(i), 0.25f, float(i % 3)}, "resize resets history");
		pass("resize rebuilds and resets history");
		test.state->SetAccumulate(false);
		test.ResetReference();
		for (unsigned i = 0; i < 6; ++i)
			test.Render({float(i), float(i + 2), 0.0f}, "accumulation off");
		pass("accumulation off displays current frame only");
		test.state->SetAccumulate(true);
		test.ResetReference();
		for (unsigned i = 0; i < 9; ++i)
			test.Render({float(i + 1), 0.5f, 0.125f}, "re-enable starts fresh window");
		pass("re-enable does not reuse disabled-frame history");
		test.SetWindow(0);
		for (unsigned i = 0; i < 20; ++i)
			test.Render({float(i % 4), 0.5f, 1.0f}, "unlimited cumulative mean");
		pass("window zero retains cumulative averaging");
		test.SetWindow(1);
		test.Render({0.25f, 1.0f, 2.0f}, "unlimited to window one");
		pass("unlimited to finite window resets history");
		Check(device->WaitIdle(), "final wait");
		Require(g_validation_errors.load() == 0, "Vulkan reported validation errors");
		std::cout << "PASS all " << test.AssertedFrames() << " GPU frames (every RGB channel of every pixel)\n";
		report << "PASS all " << test.AssertedFrames() << " GPU frames\n";
		if (validation)
			report << "PASS zero Vulkan validation errors\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "FAIL " << error.what() << '\n';
		report << "FAIL " << error.what() << '\n';
		return 1;
	}
}
