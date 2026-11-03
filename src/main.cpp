#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>

#include <spdlog/spdlog.h>

#include <charconv>
#include <cmath>
#include <string_view>
#include "RenderCapture.hpp"
#include "SceneCatalog.hpp"

#include "rg/NRCRenderGraph.hpp"

constexpr uint32_t kFrameCount = 3, kWidth = 1280, kHeight = 720;

static void ResetSceneCamera(Camera &camera, const SceneCatalogEntry &entry) {
	camera = Camera{};
	if (entry.camera) {
		const auto &preset = *entry.camera;
		camera.position = {preset.position[0], preset.position[1], preset.position[2]};
		const glm::dvec3 delta{double(preset.target[0]) - preset.position[0],
		                      double(preset.target[1]) - preset.position[1],
		                      double(preset.target[2]) - preset.position[2]};
		const auto direction = glm::normalize(delta);
		camera.yaw = static_cast<float>(std::atan2(-direction.x, -direction.z));
		camera.pitch = static_cast<float>(std::asin(std::clamp(direction.y, -1.0, 1.0)));
		camera.fov = glm::radians(preset.fov_degrees);
	} else if (entry.is_cornell) {
		camera.position = {0.0f, -0.005007153f, 3.86123033f};
		camera.fov = glm::radians(39.30765f);
	}
}

int main(int argc, char **argv) {
	const auto print_usage = [] {
		spdlog::info("Usage: VkNRC [scene.obj] [--camera x y z] [--fov degrees] [--max-bounces 1..64] "
		             "[--accumulate-last samples] [--no-nee] [--compare output-directory] [--warmup frames] [--samples frames] [--seed number]");
		spdlog::info("Without a scene argument, opens the bundled Cornell box. Choose other assets in the Scene list.");
		spdlog::info("Accumulate last: 0 = all samples (default); 1..64 = rolling window, rendering continues. "
		             "--max-spp is an alias. In --compare, --samples sets the frames rendered per capture.");
		spdlog::info("Next event estimation is enabled by default; --no-nee disables it for comparison.");
	};
	if (argc == 2 && std::string_view{argv[1]} == "--help") {
		print_usage();
		return EXIT_SUCCESS;
	}
	auto camera = myvk::MakePtr<Camera>();
	std::optional<std::filesystem::path> custom_scene;
	glm::vec3 camera_position{};
	float camera_fov{};
	bool override_position = false, override_fov = false;
	std::filesystem::path comparison_directory;
	uint32_t warmup_frames = 2048, sample_frames = 1024, comparison_seed = 1337;
	uint32_t max_bounces = VkNRCState::GetDefaultMaxBounces();
	uint32_t accumulation_window = 0;
	bool use_nee = true;
	const auto parse_uint = [](const char *text, uint32_t &value) {
		std::string_view input{text};
		auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
		return error == std::errc{} && end == input.data() + input.size();
	};
	const auto parse_float = [](const char *text, float &value) {
		std::string_view input{text};
		auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
		return error == std::errc{} && end == input.data() + input.size() && std::isfinite(value);
	};
	for (int i = 1; i < argc;) {
		std::string_view option{argv[i++]};
		if (option == "--camera" && i + 3 <= argc && parse_float(argv[i], camera_position.x) &&
		    parse_float(argv[i + 1], camera_position.y) && parse_float(argv[i + 2], camera_position.z)) {
			i += 3;
			override_position = true;
		} else if (option == "--fov" && i < argc) {
			float degrees;
			if (!parse_float(argv[i++], degrees) || degrees <= 0.0f || degrees >= 179.0f) {
				spdlog::error("Field of view must be a finite number between 0 and 179 degrees");
				return EXIT_FAILURE;
			}
			camera_fov = glm::radians(degrees);
			override_fov = true;
		} else if (option == "--max-bounces" && i < argc) {
			if (!parse_uint(argv[i++], max_bounces) || max_bounces < 1 ||
			    max_bounces > VkNRCState::GetMaxSupportedBounces()) {
				spdlog::error("Max bounces must be an integer from 1 to {}", VkNRCState::GetMaxSupportedBounces());
				return EXIT_FAILURE;
			}
		} else if ((option == "--accumulate-last" || option == "--max-spp") && i < argc) {
			if (!parse_uint(argv[i++], accumulation_window) || accumulation_window > 64) {
				spdlog::error("Accumulation window must be an integer from 0 to 64 (0 = all samples)");
				return EXIT_FAILURE;
			}
		} else if (option == "--no-nee") {
			use_nee = false;
		} else if (option == "--compare" && i < argc) {
			comparison_directory = argv[i++];
		} else if ((option == "--warmup" || option == "--samples" || option == "--seed") && i < argc) {
			uint32_t &value = option == "--warmup" ? warmup_frames : option == "--samples" ? sample_frames : comparison_seed;
			if (!parse_uint(argv[i++], value) || (option != "--seed" && value == 0)) {
				spdlog::error("Invalid integer value for {}", option);
				return EXIT_FAILURE;
			}
		} else if (!option.starts_with('-') && !custom_scene) {
			custom_scene = std::filesystem::path{option};
		} else {
			spdlog::error("Invalid option or missing numeric arguments: {}", option);
			print_usage();
			return EXIT_FAILURE;
		}
	}
	auto scene_catalog = DiscoverSceneCatalog(argv[0], custom_scene);
	auto initial_scene = custom_scene ? scene_catalog.custom_index : scene_catalog.default_index;
	if (!initial_scene) {
		spdlog::error("{}", scene_catalog.diagnostic);
		return EXIT_FAILURE;
	}
	if (!scene_catalog.diagnostic.empty())
		spdlog::warn("{}", scene_catalog.diagnostic);
	size_t scene_index = *initial_scene;
	ResetSceneCamera(*camera, scene_catalog.entries[scene_index]);
	if (override_position)
		camera->position = camera_position;
	if (override_fov)
		camera->fov = camera_fov;
	Scene scene = Scene::LoadOBJSingleInstance(scene_catalog.entries[scene_index].path);
	if (scene.Empty())
		return EXIT_FAILURE;
	spdlog::info("Scene: {}", scene_catalog.entries[scene_index].path.string());
	spdlog::info("Loaded {} Vertices, {} Texcoords, {} Materials, {} Instances", scene.GetVertices().size(),
	             scene.GetTexcoords().size(), scene.GetMaterials().size(), scene.GetInstances().size());
	GLFWwindow *window = myvk::GLFWCreateWindow("VkNRC", kWidth, kHeight, comparison_directory.empty());

	auto instance = myvk::Instance::CreateWithGlfwExtensions();
	myvk::Ptr<myvk::Queue> generic_queue, compute_queue;
	myvk::Ptr<myvk::PresentQueue> present_queue;
	myvk::Ptr<myvk::PhysicalDevice> physical_device;
	for (const auto &candidate_physical_device : myvk::PhysicalDevice::Fetch(instance)) {
		if (candidate_physical_device->GetExtensionSupport(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
			physical_device = candidate_physical_device;
			break;
		}
	}
	if (physical_device == nullptr) {
		spdlog::error("No Device Supporting VK_NV_cooperative_matrix Found");
		return -1;
	}
	spdlog::info("Physical Device: {}", physical_device->GetProperties().vk10.deviceName);
	if (!physical_device->GetFeatures().vk12.shaderSampledImageArrayNonUniformIndexing) {
		spdlog::error("Device '{}' lacks shaderSampledImageArrayNonUniformIndexing, required for per-ray material "
		              "textures. Update the GPU driver or use a GPU supporting this Vulkan 1.2 feature.",
		              physical_device->GetProperties().vk10.deviceName);
		return EXIT_FAILURE;
	}
	auto features = physical_device->GetDefaultFeatures();
	features.vk11.storageBuffer16BitAccess = VK_TRUE;
	features.vk12.bufferDeviceAddress = VK_TRUE;
	features.vk12.hostQueryReset = VK_TRUE;
	features.vk12.shaderFloat16 = VK_TRUE;
	features.vk12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	features.vk12.vulkanMemoryModel = VK_TRUE;
	features.vk12.vulkanMemoryModelDeviceScope = VK_TRUE;
	features.vk13.computeFullSubgroups = VK_TRUE;
	features.vk13.subgroupSizeControl = VK_TRUE;
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features = {
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
	    .shaderBufferFloat32AtomicAdd = VK_TRUE};
	VkPhysicalDeviceCooperativeMatrixFeaturesNV cooperative_matrix_features = {
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV,
	    .pNext = &atomic_float_features,
	    .cooperativeMatrix = VK_TRUE,
	    .cooperativeMatrixRobustBufferAccess = VK_FALSE};
	VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
	    .pNext = &cooperative_matrix_features,
	    .rayQuery = VK_TRUE};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features = {
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
	    .pNext = &ray_query_features,
	    .accelerationStructure = VK_TRUE};
	features.vk13.pNext = &accel_features;
	auto device = myvk::Device::Create(
	    physical_device,
	    myvk::GenericPresentQueueSelector{&generic_queue, myvk::Surface::Create(instance, window), &present_queue},
	    features,
	    {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	     VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
	     VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME});
	myvk::ImGuiInit(window, myvk::CommandPool::Create(generic_queue));

	Camera::Control cam_control{.sensitivity = 0.005f, .speed = 0.5f, .prev_cursor_pos = {}};

	auto vk_scene = myvk::MakePtr<VkScene>(generic_queue, scene);
	spdlog::info("Environment intensity: {} ({})", vk_scene->GetEnvironmentIntensity(),
	             vk_scene->HasEmissiveSurfaces() ? "emissive surface present" : "no emissive surfaces");
	auto vk_scene_blas = myvk::MakePtr<VkSceneBLAS>(vk_scene);
	auto vk_scene_tlas = myvk::MakePtr<VkSceneTLAS>(vk_scene_blas);
	auto vk_nrc_state = myvk::MakePtr<VkNRCState>(generic_queue, VkExtent2D{kWidth, kHeight});
	vk_nrc_state->SetMaxBounces(max_bounces);
	vk_nrc_state->SetUseNEE(use_nee);
	vk_nrc_state->SetAccumulationWindow(accumulation_window);
	if (accumulation_window != 0) {
		if (vk_nrc_state->GetAccumulationWindow() != accumulation_window) {
			spdlog::error("Could not create the requested rolling accumulation window");
			return EXIT_FAILURE;
		}
		vk_nrc_state->SetAccumulate(true);
		spdlog::info("Rolling accumulation: latest {} samples; rendering continues", accumulation_window);
	}
	spdlog::info("Max bounces: {} (explicit surface vertices, including the primary hit)", max_bounces);
	spdlog::info("Next event estimation: {}", use_nee ? "enabled" : "disabled");
	const bool compare = !comparison_directory.empty();
	if (compare) {
		vk_nrc_state->SetRandomSeed(comparison_seed);
		vk_nrc_state->ResetMLPBuffers();
		// Reproduce the interactive NRC/plain-PT split while training. Both
		// measured captures then cover the entire image using frozen weights.
		vk_nrc_state->SetRightMethod(VkNRCState::kNone);
		spdlog::info("Comparison: split-view train {} frames, then render {} frames per method, seed {}",
		             warmup_frames, sample_frames, comparison_seed);
	}
	uint32_t comparison_phase = 0, phase_frames = 0;
	int exit_code = EXIT_SUCCESS;

	auto frame_manager = myvk::FrameManager::Create(generic_queue, present_queue, false, kFrameCount);
	frame_manager->SetResizeFunc([&](VkExtent2D extent) {
		vk_nrc_state->ResetAccumulateImage(extent);
		vk_nrc_state->ResetAccumulateCount();
	});
	std::array<myvk::Ptr<rg::NRCRenderGraph>, kFrameCount> render_graphs;
	const auto rebuild_render_graphs = [&] {
		for (auto &graph : render_graphs)
			graph = myvk::MakePtr<rg::NRCRenderGraph>(frame_manager, vk_scene_tlas, vk_nrc_state, camera);
	};
	rebuild_render_graphs();
	std::array<myvk::Ptr<rg::PausedRenderGraph>, kFrameCount> paused_graphs;
	myvk::Ptr<myvk::ImageView> last_rendered_image;
	uint32_t last_rendered_spp = 0;
	bool paused = false;
	std::string scene_error;
	int view_max_bounces = static_cast<int>(max_bounces);

	bool view_accumulate = vk_nrc_state->IsAccumulate();
	bool view_use_nee = vk_nrc_state->IsUseNEE();
	int view_left_method = static_cast<int>(vk_nrc_state->GetLeftMethod());
	int view_right_method = static_cast<int>(vk_nrc_state->GetRightMethod());
	bool nrc_use_ema = vk_nrc_state->IsUseEMAWeights(), nrc_lock = false, nrc_train_one_frame = false;

	double prev_time = glfwGetTime();
	while (!glfwWindowShouldClose(window)) {
		double delta;
		{
			double cur_time = glfwGetTime();
			delta = cur_time - prev_time;
			prev_time = cur_time;
		}
		if (paused)
			glfwWaitEventsTimeout(1.0 / 30.0);
		else
			glfwPollEvents();

		nrc_train_one_frame = false;
		std::optional<size_t> requested_scene;

		myvk::ImGuiNewFrame();
		ImGui::Begin("Panel");
		ImGui::Text("FPS %.1f", ImGui::GetIO().Framerate);
		if (!compare) {
			if (ImGui::BeginCombo("Scene", scene_catalog.entries[scene_index].label.c_str())) {
				for (size_t i = 0; i < scene_catalog.entries.size(); ++i) {
					const bool selected = i == scene_index;
					if (ImGui::Selectable(scene_catalog.entries[i].label.c_str(), selected) && !selected)
						requested_scene = i;
					if (selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Lists OBJ files under assets, plus a scene supplied on the command line.\n"
				                  "Switching scenes resets the camera and cache and resumes rendering.");
			if (!scene_error.empty())
				ImGui::TextWrapped("%s", scene_error.c_str());
			if (last_rendered_image && ImGui::Button(paused ? "Resume rendering" : "Pause rendering")) {
				paused = !paused;
				if (!paused)
					glfwGetCursorPos(window, &cam_control.prev_cursor_pos.x, &cam_control.prev_cursor_pos.y);
				spdlog::info("Rendering {} at {} accumulated samples", paused ? "paused" : "resumed",
				             paused ? last_rendered_spp : vk_nrc_state->GetAccumulateCount());
			}
			if (paused) {
				ImGui::TextUnformatted("Paused: image and training frozen");
				ImGui::Text("SPP %u", last_rendered_spp);
			} else {
				if (ImGui::Checkbox("Next event estimation", &view_use_nee) && vk_nrc_state->SetUseNEE(view_use_nee)) {
					frame_manager->WaitIdle();
					vk_nrc_state->ResetMLPBuffers();
					nrc_lock = false;
				}
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Sample lights explicitly and combine with BSDF samples using MIS.\n"
					                  "Changing this restarts accumulation and cache training.");
				if (ImGui::SliderInt("Max bounces", &view_max_bounces, 1,
				                     static_cast<int>(VkNRCState::GetMaxSupportedBounces()))) {
					view_max_bounces = std::clamp(view_max_bounces, 1,
					                              static_cast<int>(VkNRCState::GetMaxSupportedBounces()));
					if (vk_nrc_state->SetMaxBounces(static_cast<uint32_t>(view_max_bounces))) {
						frame_manager->WaitIdle();
						vk_nrc_state->ResetMLPBuffers();
						nrc_lock = false;
					}
				}
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Maximum explicitly traced surface vertices, including the primary hit.\n"
					                  "1 shows visible emission/background only. Changing this resets training.\n"
					                  "NRC can approximate further transport through earlier cache queries.");
				int view_window = static_cast<int>(vk_nrc_state->GetAccumulationWindow());
				if (ImGui::InputInt("Accumulate last", &view_window, 1, 4)) {
					view_window = std::clamp(view_window, 0, static_cast<int>(vk_nrc_state->GetMaxAccumulationWindow()));
					if (static_cast<uint32_t>(view_window) != vk_nrc_state->GetAccumulationWindow()) {
						frame_manager->WaitIdle();
						vk_nrc_state->SetAccumulationWindow(static_cast<uint32_t>(view_window));
						if (vk_nrc_state->GetAccumulationWindow() != 0) {
							view_accumulate = true;
							vk_nrc_state->SetAccumulate(true);
						}
					}
				}
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Average the latest N samples per pixel while rendering continues.\n"
					                  "0 = all samples. Changing N clears history and keeps the trained cache.\n"
					                  "Maximum at this resolution: %u samples.", vk_nrc_state->GetMaxAccumulationWindow());
				if (vk_nrc_state->IsAccumulate() && vk_nrc_state->GetAccumulationWindow() != 0)
					ImGui::Text("Rolling SPP %u / %u", vk_nrc_state->GetAccumulateCount(), vk_nrc_state->GetAccumulationWindow());
				if (ImGui::Button("Restart accumulation")) {
					view_accumulate = true;
					vk_nrc_state->SetAccumulate(true);
					vk_nrc_state->ResetAccumulateCount();
				}
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Start again at 0 SPP, keeping the camera, rendering methods and trained cache.");
			}
		}
		if (compare) {
			const char *phase = comparison_phase == 0 ? "Training" : comparison_phase == 1 ? "NRC capture" : "Path tracing capture";
			ImGui::Text("%s: %u frames", phase, phase_frames);
		}
		if (!compare && !paused && ImGui::CollapsingHeader("View")) {
			if (ImGui::Checkbox("Accumulate", &view_accumulate)) {
				vk_nrc_state->SetAccumulate(view_accumulate);
			}
			if (vk_nrc_state->IsAccumulate()) {
				ImGui::SameLine();
				ImGui::Text("SPP %d", vk_nrc_state->GetAccumulateCount());
			}
			constexpr const char *kViewTypeComboStr = "None\0NRC\0Cache\0";
			if (ImGui::Combo("Left", &view_left_method, kViewTypeComboStr)) {
				vk_nrc_state->SetLeftMethod(static_cast<VkNRCState::Method>(view_left_method));
				vk_nrc_state->ResetAccumulateCount();
			}
			if (ImGui::Combo("Right", &view_right_method, kViewTypeComboStr)) {
				vk_nrc_state->SetRightMethod(static_cast<VkNRCState::Method>(view_right_method));
				vk_nrc_state->ResetAccumulateCount();
			}
		}
		if (!compare && !paused && ImGui::CollapsingHeader("NRC")) {
			if (ImGui::Checkbox("Use EMA", &nrc_use_ema)) {
				vk_nrc_state->SetUseEMAWeights(nrc_use_ema);
				vk_nrc_state->ResetAccumulateCount();
			}
			if (ImGui::Checkbox("Lock", &nrc_lock))
				vk_nrc_state->ResetAccumulateCount();
			if (nrc_lock) {
				ImGui::SameLine();
				if (ImGui::Button("Train 1-Frame")) {
					vk_nrc_state->ResetAccumulateCount();
					nrc_train_one_frame = true;
				}
			}
			if (ImGui::Button("Re-Train")) {
				frame_manager->WaitIdle();
				vk_nrc_state->ResetAccumulateCount();
				vk_nrc_state->ResetMLPBuffers();
			}
		}
		ImGui::End();
		ImGui::Render();

		if (requested_scene) {
			const auto &entry = scene_catalog.entries[*requested_scene];
			auto next_scene = Scene::LoadOBJSingleInstance(entry.path);
			if (next_scene.Empty()) {
				scene_error = "Could not load " + entry.label + "; see the log for details.";
			} else {
				frame_manager->WaitIdle();
				paused_graphs = {};
				last_rendered_image.reset();
				paused = false;
				scene = std::move(next_scene);
				vk_scene = myvk::MakePtr<VkScene>(generic_queue, scene);
				vk_scene_blas = myvk::MakePtr<VkSceneBLAS>(vk_scene);
				vk_scene_tlas = myvk::MakePtr<VkSceneTLAS>(vk_scene_blas);
				vk_nrc_state->ResetMLPBuffers();
				vk_nrc_state->ResetAccumulateCount();
				ResetSceneCamera(*camera, entry);
				glfwGetCursorPos(window, &cam_control.prev_cursor_pos.x, &cam_control.prev_cursor_pos.y);
				rebuild_render_graphs();
				scene_index = *requested_scene;
				scene_error.clear();
				nrc_lock = false;
				spdlog::info("Scene: {}; environment intensity: {}", entry.path.string(), vk_scene->GetEnvironmentIntensity());
				// Scene loading is synchronous; do not use that delay for camera movement.
				prev_time = glfwGetTime();
				delta = 0.0;
			}
		}

		if (!compare && !paused && camera->DragControl(window, &cam_control, delta))
			vk_nrc_state->ResetAccumulateCount();

		if ((compare && comparison_phase != 0) || (nrc_lock && !nrc_train_one_frame))
			vk_nrc_state->SetTrainProbability(0.0f);
		else
			vk_nrc_state->SetTrainProbability(VkNRCState::GetDefaultTrainProbability());

		if (frame_manager->NewFrame()) {
			const auto &command_buffer = frame_manager->GetCurrentCommandBuffer();
			const bool display_only = paused;
			if (display_only && !paused_graphs[0]) {
				// Finish all submitted samples before displaying the retained image.
				frame_manager->WaitIdle();
				for (auto &graph : paused_graphs)
					graph = myvk::MakePtr<rg::PausedRenderGraph>(frame_manager, last_rendered_image);
			} else if (!display_only && paused_graphs[0]) {
				frame_manager->WaitIdle();
				paused_graphs = {};
			}

			command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			if (display_only) {
				auto &graph = paused_graphs[frame_manager->GetCurrentFrame()];
				graph->SetCanvasSize(frame_manager->GetExtent());
				graph->CmdExecute(command_buffer);
			} else {
				auto &graph = render_graphs[frame_manager->GetCurrentFrame()];
				graph->SetCanvasSize(frame_manager->GetExtent());
				graph->CmdExecute(command_buffer);
				last_rendered_image = vk_nrc_state->GetAccumulateImageView();
			}
			command_buffer->End();

			// Record this sample before presentation: Render may resize the
			// accumulation image and reset its count, which must remain zero.
			if (!display_only) {
				vk_nrc_state->NextFrame();
				last_rendered_spp = vk_nrc_state->GetAccumulateCount();
			}
			frame_manager->Render();
			if (compare && ++phase_frames == (comparison_phase == 0 ? warmup_frames : sample_frames)) {
				frame_manager->WaitIdle();
				if (comparison_phase != 0) {
					auto filename = comparison_directory / (comparison_phase == 1 ? "nrc.pfm" : "path-tracing.pfm");
					if (!CaptureAccumulationPFM(generic_queue, vk_nrc_state->GetAccumulateImageView(), filename)) {
						exit_code = EXIT_FAILURE;
						break;
					}
				}
				if (++comparison_phase == 3)
					break;
				phase_frames = 0;
				vk_nrc_state->SetAccumulate(true);
				vk_nrc_state->ResetAccumulateCount();
				vk_nrc_state->SetRandomSeed(comparison_seed);
				auto method = comparison_phase == 1 ? VkNRCState::kNRC : VkNRCState::kNone;
				vk_nrc_state->SetLeftMethod(method);
				vk_nrc_state->SetRightMethod(method);
				view_left_method = view_right_method = static_cast<int>(method);
				view_accumulate = true;
			}
		}
	}

	frame_manager->WaitIdle();
	glfwTerminate();
	return exit_code;
}
