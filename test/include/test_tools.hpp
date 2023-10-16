#pragma once

#include <einops.hpp>
#include <packing.hpp>
using namespace einops;
using namespace einops::backends;
using namespace einops::implementation;

#if defined (EINOPS_TORCH_BACKEND)

#include <backends/torch_backend.hpp>

using Tensors = std::vector<torch::Tensor>;

template<typename Array, std::size_t... I>
auto array2tuple_impl(const Array& a, std::index_sequence<I...>)
{
	return std::make_tuple(a[I]...);
}

template<typename T, std::size_t N, typename Indices = std::make_index_sequence<N>>
auto array2tuple(const std::array<T, N>& a)
{
	return array2tuple_impl(a, Indices{});
}

template <size_t N = 2>
inline auto split(torch::Tensor const& tensor)
{
	std::array<torch::Tensor, N> array;
	for (int64_t n : iters::range(N))
		array[n] = tensor.index({ n });
	return array2tuple(array);
}

inline auto arange_and_reshape(int64_t arange, at::IntArrayRef const& reshape)
{
	return torch::arange(arange).reshape(reshape);
}

inline auto pack_t(std::vector<torch::Tensor> const& tensors, std::string const& pattern)
{
	return std::get<0>(pack(tensors, pattern));
}

inline auto stack(at::TensorList const& tensors, int64_t axis)
{
	return torch::stack(tensors, axis);
}

inline auto concatenate(at::TensorList const& tensors, int64_t axis)
{
	return torch::concat(tensors, axis);
}

inline auto concatenate_i(std::vector<torch::Tensor> const& tensors, int64_t axis)
{
	return torch::concat({ tensors[0].index({ Slice(None, None, None), Slice(None, None, None), None }), 
						   tensors[1].index({ Slice(None, None, None), Slice(None, None, None), None }), 
						   tensors[2].index({ Slice(None, None, None), Slice(None, None, None), None }), 
						   tensors[3] }, axis);
}

inline auto zeros(at::IntArrayRef const& size)
{
	return torch::zeros(size);
}

inline auto rand(at::IntArrayRef const& rnd)
{
	return torch::rand(rnd);
}

inline auto random(at::IntArrayRef const& rnd)
{
	return torch::randn(rnd);
}

inline auto randoms(std::size_t n, at::IntArrayRef const& rnd)
{
    std::vector<torch::Tensor> images; images.reserve(n);
    std::generate_n(std::back_inserter(images), n, [=]() mutable { return torch::randn(rnd); });
    return images;
}

inline auto substract(torch::Tensor const& x, torch::Tensor const& y)
{
	return x - y;
}

inline std::string dump(torch::IntArrayRef const& vector)
{
	std::string result = "(";
	for (auto value : vector)
		result += std::to_string(value) + ", ";
	result = result.substr(0, result.size() - 2);
	return result + ")";
}

inline std::string dump(torch::Tensor const& tensor)
{
	return dump(tensor.sizes());
}

inline bool array_equal(torch::Tensor const& lhs, torch::Tensor const& rhs)
{
	return dump(lhs.sizes()) == dump(rhs.sizes());
}

inline auto comp_type(torch::Tensor const& lhs, torch::Tensor const& rhs) -> bool
{
	return (lhs.dtype() == rhs.dtype());
}

inline auto comp_shape(torch::Tensor const& lhs, torch::Tensor const& rhs) -> bool
{
	return (lhs.sizes() == rhs.sizes());
}

inline auto comp_all(torch::Tensor const& lhs, torch::Tensor const& rhs) -> bool
{
	return (lhs.toString() == rhs.toString());
}

inline auto comp_allclose(torch::Tensor const& lhs, torch::Tensor const& rhs) -> bool
{
	return (lhs.toString() == rhs.toString());
}

#endif // EINOPS_BACKEND

template <typename T>
inline std::string dump(std::initializer_list<T> const& values)
{
	std::string result = "(";
	for (auto value : values)
		result += std::to_string(value) + ", ";
	result = result.substr(0, result.size() - 2);
	return result + ")";
}

template <typename T>
inline std::string dump(std::vector<T> const& values)
{
	std::string result = "(";
	for (auto value : values)
		result += dump(value) + ", ";
	result = result.substr(0, result.size() - 2);
	return result + ")";
}

inline std::string dump_map(std::map<std::string, int64_t> const& values)
{
	std::string result = "{";
	for (auto&& [k, v] : values)
		result += "'" + k + "': " + std::to_string(v) + ", ";
	result = result.substr(0, result.size() - 2);
	return result + "}";
}

class Timer
{
	using clock = std::chrono::steady_clock;
	clock::time_point start_time = {};
	clock::duration elapsed_time = {};

public:
	bool running() const 
	{
		return start_time != clock::time_point{};
	}

	void start()
	{
		if (!running())
		{
			start_time = clock::now();
		}
	}

	void stop()
	{
		if (running())
		{
			elapsed_time += clock::now() - start_time;
			start_time = {};
		}
	}

	void reset() 
	{
		start_time = {};
		elapsed_time = {};
	}

	clock::duration get() 
	{
		auto result = elapsed_time;
		if (running())
			result += clock::now() - start_time;
		return result;
	}
};

class UnitTest 
{
public:
	UnitTest(std::string const& name)
		: name(name)
		, duration(0)
	{}

	template <typename T> 
	void check(T a, T b, std::string const& stra, 
						 std::string const& strb, 
						 std::string const& file, int line, 
						 std::string const& func)
	{
		TESTs++; 
		if (a == b) 
		{ 
			//std::cout << "."; 
			return;
		}

		fails++; 
		//std::cout << "F"; 

		serr << single_line << std::endl;
		serr << " CODE: " << func << std::endl;
		serr << " FILE: \"" << file << "\", line " << line << std::endl;
		serr << " TEST: " << stra << " == " << strb << std::endl;
		serr << " FAIL: " << a << " != " << b << std::endl;
	}

	int status() 
	{
		if (fails)
			std::cout << "FAILED (failures=" << fails << ")" << std::endl;
		else
			std::cout << "PASSED" << std::endl;
		std::cout << "Running " << TESTs << " tests in " << format_duration() << std::endl;
		if (fails) std::cout << serr.str();
		return fails > 0;
	}

	virtual void test_list() = 0;

	int run()
	{
		std::cout << single_line << std::endl;
		std::cout << "Testing '" << name << "': ";
		Timer timer;
		timer.start();
		{
			test_list();
		}
		timer.stop();
		duration = timer.get();
		return status();
	}

private:
	std::string name;
	int TESTs{ 0 }, fails{ 0 };
	std::ostringstream serr;
	std::chrono::steady_clock::duration duration;

	auto format_duration() -> std::string
	{
		return format("{} s ({} ms)", double(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()) / 1000.0,
									  double(std::chrono::duration_cast<std::chrono::microseconds>(duration).count()) / 1000.0);
	}

	const int ncols { 80 };
	const std::string single_line = std::string(ncols, '-');
	const std::string double_line = std::string(ncols, '=');

public:
#define TESTS(a,b) check<std::string>(a, b, #a, #b, __FILE__, __LINE__, __FUNCTION__);
#define TESTB(a)   check<bool>(a, true, #a, "true", __FILE__, __LINE__, __FUNCTION__);
};