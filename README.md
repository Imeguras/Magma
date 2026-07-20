# Rocm's Magma!
Tired of watching proprietary nvidia scooping up the computer vision scene, with proprietary frameworks like DeepStream? Watch as the rocm melts into streaming molten magma!
To put it bluntly the idea is not doing **the whole compatibility and flexibility over performance**, the reason is: 
If nvidia won't play ball so whats the point of making a software more complex, if the bigger fish reaps the rewards of deep optimization?

## INB4's
Q: Well is this a dead project/wheres results?
A: Im known to easily distract myself but heres a proof that it does... well something!
Its working in a 7800xt(neofetch misread it due to pcie identifiers), its a bit choppy since the gpu is already being hamered by a bunch of apps(obs, vscode) and i wasnt responsible with mgmosd and mgmtensordump(i mostly vibe coded them to speed up development):
![Inference on a 7800xt](assets/inference.gif)

## AI Usage

Permited as long as it doesnt become mindless pushing and you are using it as an actual tool(ie you are actually thinking and trying to upkeep code). However **ignore** all file artifacts relating to agents, AI-related markdowns, etc... 
Ie some vibecoding is allowed but only to speed up some development ie core stuff should be reread by hand

## License

This code is provided under a LGPLv3 license check LICENSE.md

## See it in action!
### Docker
Theres not much to say, its just an example to ease developing over the libraries, theres a docker-compose file to pull the full playtest environment.

### assuming you already have the environment setup
**I know it seems like a lot** but 1/2 of the commands are just getting the onnx(i tested with some other models but for some reason no results and im too lazy to debug when from ultralytics it works).

```sh
#fetch the model:
curl -L -o /tmp/yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt" 
python -c "import shutil, os; os.makedirs('./gst-plugin/tests/onnx-gen/onnx-models', exist_ok=True); shutil.move('/tmp/yolov8n.onnx', './gst-plugin/tests/onnx-gen/onnx-models/yolov8n.onnx')"
#compile the onnx into a mxr(straight to onnx is also possible but pathwise its annoying regardless of how it would be implemented so... its more convenient to just organise it to your liking)
migraphx-driver compile --onnx ./gst-plugin/tests/onnx-gen/onnx-models/yolov8n.onnx --binary --output ./gst-plugin/tests/onnx-gen/migraph/yolov8n.mxr
#watch the magic happen on rocm's environment
gst-launch-1.0 filesrc location=./gst-plugin/tests/test_data/ny-walking.mp4 ! qtdemux ! h264parse ! vah264dec ! mgmvideoconvert \
  ! mgmpreproc net-width=640 net-height=640 enable-roi=true roi-x=0 roi-y=587 roi-w=719 roi-h=451 \
  ! mgminfer model-onnx-file=./gst-plugin/tests/onnx-gen/onnx-models/yolov8n.onnx model-mxr-file=./gst-plugin/tests/onnx-gen/migraph/yolov8n.mxr \
      parser-plugin=/usr/lib/magma/addons/libyolov8-parser.so \
      confidence-threshold=0.51 nms-threshold=0.45 max-detections=1000 \ 
  ! mgmosd \
  ! mgmvideoconvert ! autovideosink

```

But yolo is for ner...
```sh
#ok this one is actually more difficult as i thought that ai nerds would share their .onnx(tipically where i work getting the .onnx is a matter of just annoying the person) and i struggled so imma write for arch linux 
cp ./gst-plugin/tests/onnx-gen/scaffolds/biformer_tiny.py /tmp/pit.py

cd /tmp
git clone https://github.com/rayleizhu/BiFormer.git
cd BiFormer
mv /tmp/pit.py ./pit.py
# use whatever you like but we all know paru is superior
paru -S python312 python312-virtualenv execstack patchelf
#i hate python so yes venv is good enough for me
python3.12 -m venv BiFormer-env
source BiFormer-env/bin/activate
#the deps for getting the ducking onnx
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install timm einops fairscale onnx
#Ai slop time, apparently the .so's were compiled wrong so we're patching them. If you ask me id just bruteforce this because this is literally why i hate AI development nothing works outside their pc but they are forgiven for posting good numbers(meanwhile i get shat on when some network is misconfigured on docker).
sudo execstack -c /tmp/BiFormer/BiFormer-env/lib/python3.12/site-packages/pytorch_triton_rocm/lib/libamdhip64.so*
sudo execstack -c /tmp/BiFormer/BiFormer-env/lib/python3.12/site-packages/torch/lib/libamdhip64.so
patchelf --clear-execstack /tmp/BiFormer/BiFormer-env/lib/python3.12/site-packages/torch/lib/libamdhip64.so
find /tmp/BiFormer/BiFormer-env/lib/python3.12/site-packages/torch/lib/ -name "*.so*" -exec patchelf --clear-execstack 

python3.12 pit.py
#voila you should have an .onnx! GO BACK TO MAGMA DIR!!!
READ_ME cd_into_magma_dir
mv /tmp/BiFormer/biformer_tiny.onnx ./gst-plugin/tests/onnx-gen/onnx-models/
#alright we're back to the cooking area
migraphx-driver compile --onnx ./gst-plugin/tests/onnx-gen/onnx-models/biformer_tiny.onnx --binary --output ./gst-plugin/tests/onnx-gen/migraph/biformer_tiny.mxr
gst-launch-1.0 filesrc location=./gst-plugin/tests/test_data/ny-walking.mp4 ! \
  qtdemux ! h264parse ! vah264dec ! mgmvideoconvert ! \
  mgmpreproc net-width=224 net-height=224 enable-roi=true roi-x=0 roi-y=587 roi-w=719 roi-h=451 ! \
  mgminfer model-onnx-file=./gst-plugin/tests/onnx-gen/onnx-models/biformer_tiny.onnx \
    model-mxr-file=./gst-plugin/tests/onnx-gen/migraph/biformer_tiny.mxr \
    parser-plugin=/usr/lib/magma/addons/libmagmabiformer-desc-parser.so \
    confidence-threshold=0.50 max-detections=5 ! \
  mgmosd ! mgmvideoconvert ! autovideosink
```
oh btw if you want a detector theres just use biformer-detect_tiny.py instead like so:
```sh
cp ./gst-plugin/tests/onnx-gen/scaffolds/biformer_tiny.py /tmp/pit.py
...
python3.12 pit.py
biformervit-detect_tiny.onnx -> biformervit-detect_tiny.mxr

```
## Documentation

API documentation is generated from Doxygen comments in the source code
using **Doxygen** → **Sphinx** (via Breathe).

```sh
cd docs && doxygen && python3 -m sphinx -b html . _build/html
```

Then open `docs/_build/html/index.html` in a browser.

From VS Code: press **Ctrl+Shift+P** → **Tasks: Run Task** → **Build docs**.

### how do i compile the god damn thing:
I do it like so since im on arch and /usr/local is a mess
```sh
rm -rf build && \                                                                                                                                                                                  14:32:30
meson setup build --prefix=/usr --buildtype=release && \
meson compile -C build && \
sudo meson install -C build
```
