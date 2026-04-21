#include "string"
#include "fstream"
#include "json.hpp"
#include <fstream>

bool loadConfig(const std::string &filepath, nlohmann::json &config)
{
    std::ifstream file(filepath);
    if (file.is_open())
    {
        file >> config;
        file.close();
        return true;
    }
    else
    {

        if (!std::filesystem::exists(filepath))
        {
            throw std::runtime_error("Config file does not exist: " + filepath);
        }

        if (!file.good())
        {
            throw std::runtime_error("Failed to open config file: " + filepath);
        }
        else
        {
            throw std::runtime_error("Unknown error occurred while opening config file: " + filepath);
        }
        return false;
    }
    return false;
}