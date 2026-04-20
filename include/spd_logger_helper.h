#pragma once

#include <spdlog/spdlog.h>
#include <string_view>
#include <algorithm>

inline std::string extract_function_name(const char *pretty_function)
{
    std::string_view sv(pretty_function);

    size_t end = sv.find('(');
    if (end == std::string_view::npos)
    {
        end = sv.length();
    }

    size_t start = sv.rfind(' ', end);
    if (start == std::string_view::npos)
    {
        start = 0;
    }
    else
    {
        start++;
    }

    return std::string(sv.substr(start, end - start));
}

inline std::string format_function_name(const char *pretty_function, size_t width)
{
    std::string func_name = extract_function_name(pretty_function);
    std::string result = "[" + func_name + "]";
    if (result.length() < width)
    {
        result.append(width - result.length(), ' ');
    }
    return result;
}

#define LOG_INFO(msg, ...) \
    spdlog::info("{}" msg, format_function_name(__PRETTY_FUNCTION__, 73), ##__VA_ARGS__)

#define LOG_WARN(msg, ...) \
    spdlog::warn("\033[1;33m{}" msg "\033[0m", format_function_name(__PRETTY_FUNCTION__, 70), ##__VA_ARGS__)

#define LOG_ERROR(msg, ...) \
    spdlog::error("\033[1;31m{}" msg "\033[0m", format_function_name(__PRETTY_FUNCTION__, 72), ##__VA_ARGS__)

#define LOG_DEBUG(msg, ...) \
    spdlog::debug("\033[1;34m{}" msg "\033[0m", format_function_name(__PRETTY_FUNCTION__, 72), ##__VA_ARGS__)
