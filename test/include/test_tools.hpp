#pragma once

#include <einops.hpp>
using namespace einops;
using namespace einops::backends;

#ifdef EINOPS_TORCH_BACKEND
#include <backends/torch_backend.hpp>

auto arange_and_reshape(int64_t arange, at::IntArrayRef const& reshape)
{
	return torch::arange(arange).reshape(reshape);
}

auto random(at::IntArrayRef const& rnd)
{
	return torch::randn(rnd);
}

auto build_random_images(std::size_t n, at::IntArrayRef const& rnd)
{
    std::vector<torch::Tensor> images; images.reserve(n);
    std::generate_n(std::back_inserter(images), n, [=]() mutable { return torch::randn(rnd); });
    return images;
};
#endif

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
			std::cout << "."; 
			return;
		}

		fails++; 
		std::cout << "F"; 

		serr << single_line << std::endl;
		serr << " CODE: " << func << std::endl;
		serr << " FILE: \"" << file << "\", line " << line << std::endl;
		serr << " TEST: " << stra << " == " << strb << std::endl;
		serr << " FAIL: " << a << " != " << b << std::endl;
	}

	int status() 
	{
		if (fails)
			std::cout << " FAILED (failures=" << fails << ")" << std::endl;
		else
			std::cout << " PASSED" << std::endl;
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

	const int ncols { 120 };
	const std::string single_line = std::string(ncols, '-');
	const std::string double_line = std::string(ncols, '=');

public:
#define TESTS(a,b) check<std::string>(a, b, #a, #b, __FILE__, __LINE__, __FUNCTION__);
#define TESTB(a)   check<bool>(a, true, #a, "true", __FILE__, __LINE__, __FUNCTION__);
};