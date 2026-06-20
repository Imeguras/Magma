# Rocm's Magma!

Tired of watching proprietary nvidia scooping up the computer vision scene, with proprietary frameworks like DeepStream? Watch as the rocm melts into streaming molten magma!
To put it bluntly the idea is not doing **the whole compatibility and flexibility over performance**, the reason is well nvidia won't play ball so whats the point of making a software more complex, if the bigger fish reaps the rewards of deep optimization?

Als

## AI Usage

Permited as long as it doesnt become mindless pushing and you are using it as an actual tool(ie you are actually thinking and trying to upkeep code). However **ignore** all file artifacts relating to agents, AI-related markdowns, etc...

## License

This code is provided under a LGPLv3 license check LICENSE.md

## Usage

Configure and build all examples (application and plugins) as such:

    meson builddir
    ninja -C builddir


Once the plugin is built you can either install system-wide it with `sudo ninja
-C builddir install` (however, this will by default go into the `/usr/local`
prefix where it won't be picked up by a `GStreamer` installed from packages, so
you would need to set the `GST_PLUGIN_PATH` environment variable to include or
point to `/usr/local/lib/gstreamer-1.0/` for your plugin to be found by a
from-package `GStreamer`).

Alternatively, you will find your plugin binary in `builddir/gst-plugins/src/`
as `libgstplugin.so` or similar (the extension may vary), so you can also set
the `GST_PLUGIN_PATH` environment variable to the `builddir/gst-plugins/src/`
directory (best to specify an absolute path though).

