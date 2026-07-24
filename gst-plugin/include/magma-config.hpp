#pragma once

#include <gst/gst.h>
#include <memory>
#include <string>

/**
 * @file magma-config.hpp
 * @brief Generic TOML-based configuration loader for GStreamer elements.
 *
 * Parses a TOML file and applies key-value pairs from a named section
 * as GObject properties on a GstElement.
 *
 * Config file layout:
 * @code{.toml}
 * [mgminfer]
 * model-onnx-file = "model.onnx"
 * confidence-threshold = 0.3
 * max-detections = 10
 * @endcode
 */

namespace magma {

/**
 * @brief Load and apply TOML configuration sections to GstElements.
 *
 * Typical usage inside a GStreamer element's start() method:
 * @code
 *   if (self->config_file && self->config_file[0]) {
 *       magma::Config cfg;
 *       if (cfg.load(self->config_file))
 *           cfg.apply(GST_ELEMENT(self), "mgminfer");
 *   }
 * @endcode
 */
class Config {
public:
    Config();
    ~Config();

    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&) = default;
    Config& operator=(Config&&) = default;

    /**
     * @brief Parse a TOML file from disk.
     * @param filepath Path to the .toml file
     * @return true on success
     */
    bool load(const std::string& filepath);

    /**
     * @brief Check whether a named section exists.
     */
    bool has_section(const std::string& name) const;

    /**
     * @brief Apply every key-value pair under [section] as GObject properties.
     *
     * Each TOML key is converted to a GObject property name (hyphens preserved).
     * Values are set via gst_util_set_object_arg() which handles int, float,
     * bool, string, enum, and flags automatically.
     *
     * @param element Target GstElement
     * @param section TOML section name (e.g., "mgminfer")
     * @return true if at least one property was applied
     */
    bool apply(GstElement* element, const std::string& section) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace magma
