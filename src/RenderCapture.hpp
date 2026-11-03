#pragma once

#include <filesystem>
#include <myvk/ImageView.hpp>
#include <myvk/Queue.hpp>

// Save the accumulated linear RGB radiance, before tone mapping or GUI overlays.
// The image must contain at least one accumulated frame, use RGBA32F with
// TRANSFER_SRC usage, and be in GENERAL layout on the supplied rendering queue.
// This synchronous operation waits for the device and restores GENERAL layout.
// PFM pixels are RGB float32, little endian, with the bottom row stored first.
bool CaptureAccumulationPFM(const myvk::Ptr<myvk::Queue> &queue,
                            const myvk::Ptr<myvk::ImageView> &image_view,
                            const std::filesystem::path &filename);
