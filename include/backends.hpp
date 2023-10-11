#pragma once

#ifdef EINOPS_TORCH_BACKEND
#include <backends/torch_backend.hpp>
#endif

#if !defined(EINOPS_TORCH_BACKEND) 
#error "einops-cpp: you need at least have one tensor backend included"
#endif