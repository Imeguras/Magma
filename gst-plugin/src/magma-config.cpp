#include "magma-config.hpp"
#include <cstdio>
#include <cstring>
#include <toml++/toml.hpp>

struct magma::Config::Impl {
    toml::table data;
    bool loaded = false;
};

magma::Config::Config()
    : impl_(std::make_unique<Impl>())
{
}

magma::Config::~Config() = default;

bool magma::Config::load(const std::string& filepath)
{
    try {
        impl_->data = toml::parse_file(filepath);
        impl_->loaded = true;
        return true;
    } catch (const toml::parse_error& e) {
        auto desc = std::string(e.description());
        fprintf(stderr, "magma::Config error: failed to parse %s: %s (line %u)\n",
                filepath.c_str(), desc.c_str(), e.source().begin.line);
        return false;
    }
}

bool magma::Config::has_section(const std::string& name) const
{
    return impl_->loaded && impl_->data.contains(name)
        && impl_->data[name].is_table();
}

bool magma::Config::apply(GstElement* element, const std::string& section) const
{
    if (!impl_->loaded) {
        g_warning("magma::Config: config not loaded, cannot apply section [%s]", section.c_str());
        return false;
    }

    if (!has_section(section)) {
        g_warning("magma::Config: section [%s] not found in config", section.c_str());
        return false;
    }

    const auto& tbl = *impl_->data[section].as_table();
    bool applied = false;

    for (const auto& [key, val] : tbl) {
        std::string keystr(key.str());
        std::string str;
        if (val.is_string()) {
            str = val.as_string()->get();
        } else if (val.is_floating_point()) {
            str = std::to_string(val.as_floating_point()->get());
        } else if (val.is_integer()) {
            str = std::to_string(val.as_integer()->get());
        } else if (val.is_boolean()) {
            str = val.as_boolean()->get() ? "true" : "false";
        } else {
            g_warning("magma::Config: skipping key '%s' in [%s] — unsupported TOML type",
                      keystr.c_str(), section.c_str());
            continue;
        }

        gst_util_set_object_arg(G_OBJECT(element), keystr.c_str(), str.c_str());
        applied = true;
    }

    return applied;
}
