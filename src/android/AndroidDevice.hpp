#pragma once

#include <myvk/Device.hpp>
#include <myvk/Queue.hpp>
#include <string>

myvk::Ptr<myvk::Device> CreateAndroidDevice(const myvk::Ptr<myvk::Instance> &instance, VkSurfaceKHR surface,
                                            myvk::Ptr<myvk::Queue> &queue, std::string &report,
                                            bool reference);
