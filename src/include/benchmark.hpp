#include <chrono>
#include <string>
#include <string_view>

template <typename Func>
std::string
benchmark(Func &&func, std::size_t times)
{
    using namespace std::string_view_literals;

    double min_duration = std::numeric_limits<double>::max();
    double duration = 0.0;
    // 1回分捨てる
    func();
    for (std::size_t time = 0u; time < times; ++time)
    {
        auto const start = std::chrono::system_clock::now();
        func();
        auto const end = std::chrono::system_clock::now();
        std::chrono::duration<double> dur = end - start;
        duration += dur.count();
        min_duration = std::min(min_duration, dur.count());
    }

    std::ostringstream oss;
    oss << "avg = "sv << duration / times << ", min = "sv << min_duration;
    return std::move(oss).str();
}
