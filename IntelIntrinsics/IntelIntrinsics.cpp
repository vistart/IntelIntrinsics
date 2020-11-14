// IntelIntrinsics.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>
#include <immintrin.h>

using namespace std;

double compute_pi_native(size_t dt)
{
	double pi = 0.0;
	double const delta = 1.0 / dt;
	double const squ_delta = 1.0 / (dt * dt);
	for (size_t i = 0; i < dt; i++)
	{
		double x = i * delta;
		pi += delta / (1 + x * x);
	}
	return pi * 4;
}

double compute_pi_avx256(size_t dt)
{
	double pi = 0.0;
	double const delta = 1.0 / dt;
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4;
	ymm0 = _mm256_set1_pd(1.0);
	ymm1 = _mm256_set1_pd(delta);
	ymm2 = _mm256_set_pd(delta * 3, delta * 2, delta, 0.0);
	ymm4 = _mm256_setzero_pd();
	for (int i = 0; i < dt - 4; i += 4) {
		ymm3 = _mm256_set1_pd(i * delta); // x = i * delta
		ymm3 = _mm256_add_pd(ymm3, ymm2); // x1 = x, x2 = x + delta, x3 = x + 2 * delta, x4 = x + 3 * delta
		ymm3 = _mm256_mul_pd(ymm3, ymm3); // x * x
		ymm3 = _mm256_add_pd(ymm0, ymm3); // 1 + x * x
		ymm3 = _mm256_div_pd(ymm1, ymm3); // delta / (1 + x * x)
		ymm4 = _mm256_add_pd(ymm4, ymm3); // pi += delta / (1 + x * x)
	}
	//double tmp[4] __attribute__((aligned(32)));
	double tmp[4];
	_mm256_store_pd(tmp, ymm4);
	pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
	return pi * 4.0;
}

double compute_pi_avx512(size_t dt)
{
	double pi = 0.0;
	double const delta = 1.0 / dt;
	__m512d
		ymm0 = _mm512_set1_pd(1.0),
		ymm1 = _mm512_set1_pd(delta),
		ymm2 = _mm512_set_pd(delta * 7, delta * 6, delta * 5, delta * 4, delta * 3, delta * 2, delta, 0.0),
		ymm3,
		ymm4 = _mm512_setzero_pd();
	for (int i = 0; i < dt - 8; i += 8) {
		ymm3 = _mm512_set1_pd(i * delta); // x = i * delta
		ymm3 = _mm512_add_pd(ymm3, ymm2); // x1 = x, x2 = x + delta, x3 = x + 2 * delta, x4 = x + 3 * delta, ...
		ymm3 = _mm512_mul_pd(ymm3, ymm3); // x * x
		ymm3 = _mm512_add_pd(ymm0, ymm3); // 1 + x * x
		ymm3 = _mm512_div_pd(ymm1, ymm3); // delta / (1 + x * x)
		ymm4 = _mm512_add_pd(ymm4, ymm3); // pi += delta / (1 + x * x)
	}
	double tmp[8];
	_mm512_store_pd(tmp, ymm4);
	pi += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
	return pi * 4.0;
}

int main()
{
	int times = 0;
	do
	{
		cin >> times;
	} while (times <= 0);
	vector<int> duration(times, 0);
	srand((unsigned)time(NULL));
	clock_t start, end;
	size_t dt = 2147483647;
	double result1 = 0, result2 = 0, result3 = 0;
	//普通函数计时
	for (int i = 0; i < times; i++)
	{
		start = clock();
		result1 = compute_pi_native(dt);
		end = clock();
		duration[i] = end - start;
	}
	double sum = accumulate(duration.begin(), duration.end(), 0.0);
	double mean = sum / times;
	double acc = 0;
	for_each(duration.begin(), duration.end(), [&](const double d)
	{
		acc += (d - mean) * (d - mean);
	});
	cout.flags(ios::fixed);
	cout.precision(32);
	cout << "native:\n" << "result: " << result1 << endl << "mean: " << mean << endl << "stdev: " << sqrt(acc / times) << endl;
#ifdef __AVX2__
	//avx256计时
	for (int i = 0; i < times; i++)
	{
		start = clock();
		result2 = compute_pi_avx256(dt);
		end = clock();
		duration[i] = end - start;
	}
	sum = accumulate(duration.begin(), duration.end(), 0.0);
	mean = sum / times;
	acc = 0;
	for_each(duration.begin(), duration.end(), [&](const double d)
	{
		acc += (d - mean) * (d - mean);
	});
	cout << "avx256:\n" << "result: " << result2 << endl << "mean: " << mean << endl << "stdev: " << sqrt(acc / times) << endl;
#endif
#ifdef __AVX512__
	//avx512计时
	for (int i = 0; i < times; i++)
	{
		start = clock();
		result3 = compute_pi_avx512(dt);
		end = clock();
		duration[i] = end - start;
	}
	sum = accumulate(duration.begin(), duration.end(), 0.0);
	mean = sum / times;
	acc = 0;
	cout << "avx512:\n" << "result: " << result3 << endl << "mean: " << mean << endl << "stdev: " << sqrt(acc / times) << endl;
	cout << endl;
#endif
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
