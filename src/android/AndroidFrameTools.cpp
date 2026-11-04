#include "AndroidFrameTools.hpp"

#include <cerrno>
#include <cstdlib>
#include <system_error>
#include <unistd.h>

#include <spdlog/fmt/fmt.h>

std::string AndroidFrameTools::StartBenchmark(unsigned seconds) {
	if (seconds < 1 || seconds > 120)
		return "Benchmark duration must be 1 to 120 seconds.";
	if (benchmark_.IsActive())
		return "A benchmark is already running.";
	benchmark_.Start(seconds);
	return fmt::format("Benchmark started. Warming up for {} frames, then measuring for {} s.",
	                   benchmark_.WarmupRemaining(), seconds);
}

void AndroidFrameTools::ResetFrame() {
	benchmark_.Cancel();
	last_frame_.reset();
}

std::optional<std::string> AndroidFrameTools::FinishRenderedFrame(const myvk::Ptr<myvk::Queue> &queue,
                                                                const VkNRCState &state,
                                                                std::string_view scene_name) {
	last_frame_ = state.GetAccumulateImageView();
	last_scene_.assign(scene_name);
	last_settings_ = {
	    .nee = state.IsUseNEE(),
	    .left_method = state.GetLeftMethod(),
	    .right_method = state.GetRightMethod(),
	    .max_bounces = state.GetMaxBounces(),
	    .spp = state.GetRenderedSPP(),
	    .scene_name = {},
	    .fast_mode = state.IsFastMode(),
	    .min_roughness = state.GetMinRoughness(),
	};

	VkResult drained = VK_SUCCESS;
	const auto result = benchmark_.FinishRenderedFrame([&] { drained = queue->WaitIdle(); });
	if (drained != VK_SUCCESS) {
		benchmark_.Cancel();
		return fmt::format("Benchmark failed waiting for the GPU (VkResult {}).", int(drained));
	}
	if (!result)
		return std::nullopt;
	return fmt::format("Benchmark: {} frames in {:.3f} s, {:.2f} ms/frame, {:.2f} FPS.",
	                   result->frames, result->seconds, result->MeanMilliseconds(), result->FramesPerSecond());
}

std::string AndroidFrameTools::Capture(const myvk::Ptr<myvk::Queue> &queue, const std::filesystem::path &directory,
                                      bool write_metadata) const {
	if (benchmark_.IsActive())
		return "Finish or cancel the benchmark before capturing a frame.";
	if (!last_frame_)
		return "Waiting for the first rendered frame.";
	if (directory.empty())
		return "Capture directory unavailable.";
	std::error_code error;
	std::filesystem::create_directories(directory, error);
	if (error)
		return "Cannot create capture directory: " + error.message();

	// Reserve a unique name across repeated captures, including simultaneous
	// activities. App-specific external storage requires no runtime permission.
	auto filename = (directory / "frame-XXXXXX.exr").string();
	const int fd = mkstemps(filename.data(), 4);
	if (fd < 0)
		return "Cannot create capture file: " + std::error_code(errno, std::generic_category()).message();
	const bool closed = close(fd) == 0;
	CaptureMetadata metadata{.settings = last_settings_, .write_header = write_metadata};
	metadata.settings.scene_name = last_scene_;
	if (!closed || !CaptureAccumulationEXR(queue, last_frame_, filename, &metadata)) {
		std::filesystem::remove(filename, error);
		return "Could not save the EXR frame. Check available storage and the renderer log.";
	}
	return "Saved linear EXR: " + filename;
}
