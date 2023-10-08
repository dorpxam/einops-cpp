#pragma once

#ifdef EINOPS_TORCH_BACKEND
#include <backends/torch_backend.hpp>
#endif

#ifdef EINOPS_TENSORFLOW_BACKEND
#include <backends/tensorflow_backend.hpp>
#endif

#ifndef EINOPS_TORCH_BACKEND || EINOPS_TENSORFLOW_BACKEND
#error "einops-cpp: you need at least have one tensor backend included"
#endif