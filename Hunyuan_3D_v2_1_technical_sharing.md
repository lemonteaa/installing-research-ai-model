# Hunyuan 3D 2.1

**Work in progress - Draft**

## Info card - key links

- Main huggingface model collection: https://huggingface.co/collections/tencent/hunyuan3d
- GitHub repos: https://github.com/Tencent-Hunyuan/Hunyuan3D-2 , https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1

## Background

Related to but distinct from image generation models, the text/image to 3D/SVG task is interesting in that both have two drastically different possible approach - a natively visual-spatial approach that's similar in spirit to image generation, and a neuro-symbolic approach that generates codes that expresses 3D/SVG object in a structured way. text/image to 3D is also something that would potentially unlock values in assisting 3D game assets creation, among other thing. However, it is a rather challenging modality because 1) The dataset is very expensive to create (it's very laborious even for human to create a 3D model object), and 2) the skills associated with 3D modelling (I'm totally a noob on this one) - such as meshing the geometry in a sustainable way - is subtly difficult to master (there are many way to kind of model the object but so that the mesh will cause problem down the line). There is a limited number of 3D generation AI model, and Hunyuan performs very well in my limited one-example testing (Microsoft Trellis is quite good also, TripoSR seems more limited).

Installing 3D generation model is expected to be tricky because the utilities to work with 3D objects involves niche software installations that's quite system dependent and also very *heavy* so to speak.

## Installation

### Part 1 - Basic information gathering, planning

At first glance, their GitHub repo at least is more well organized relatively speaking, with more information provided on the README. (But the model card on huggingface is barebone, so we will use the version on GitHub as the main reference) It also separates into different repo for version change (which is a good thing because the underlying model architecture may have changed). On the other hand, there is quite a few variant models and we don't have the time to test those yet, so focus only on the basic version.

First, a partial listing of the models they provided is at https://github.com/Tencent-Hunyuan/Hunyuan3D-2?tab=readme-ov-file#-models-zoo (See the huggingface model collection in the info card above for the full list). To summarize:

- **Hunyuan3D 1, Hunyuan3D 2, Hunyuan3D 2.1** - the main models
- **Hunyuan3D 2 mini** - Small version for less VRAM requirements
- **Hunyuan3D 2 mv** - finetuned from Hunyuan3D-2 to support **multiview** controlled shape generation
- **Hunyuan3D Omni** - unified framework for generation conditioned on multi-modal input. Supported input: bounding box, skeletal pose, point cloud, voxel
- **Hunyuan3D Part** - (Not sure) add a post processing pipeline to perform 3D Part segmentation (P3-SAM), followed by structure coherent shape decomposition (X-Part)

(Note: HunyuanWorld models not included in the list above)

The version 2 specifically have variants: a normal version, guidance distillation (`-Fast`), and step distillation (`-Turbo`), which are common techniques in diffusion models.

Generally, newer version of the model perform better, but the code supports may be less mature/be less feature complete.

A caveat that we immediately see from the README (applies for version 2.1 but not 2):

> It takes 10 GB VRAM for shape generation, 21GB for texture generation and 29GB for shape and texture generation in total.

So for running both shape and texture, a 24GB VRAM card is not enough and you need something like the L40S. One possible workaround? Split workload into the two step separately if it's not a on-demand workload.

They do provide a `diffuser` like pipeline interface:

```py
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
from textureGenPipeline import Hunyuan3DPaintPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# let's generate a mesh first
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh_untextured = shape_pipeline(image='assets/demo.png')[0]

paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
mesh_textured = paint_pipeline(mesh_path, image_path='assets/demo.png')
```

From above we can see that the data dependency are:

- Shape model: image (input), untextured mesh (output)
- Texture model: image + untextured mesh (input), textured mesh (output)


For version pinning, the officially provided combo are: Python 3.10, PyTorch 2.5.1, Cuda 12.4.

The provide both direct inference code, API server, as well as a gradio demo. We will go with the API server this time. Notice that the API server seems be coded by AI.

Info on the API server:

- Main code in file `/api_server.py` - can see the command line arguments, the kind of endpoints provided, the detailed logic of what it does, etc. (Notes: although the `SAVE_DIR` seems hardcoded at first glance, it's actually reassigned by `args.cache_path` later so it's okay)
- AI written doc at `/API_DOCUMENTATION.md`. Notes that both sync and async (task based) endpoints are provided.
- For further code tracing, the worker for actually doing inferencing is at `/model_worker.py`.
- Some gotchas: path issue for relative import (more on this later), endpoint currently don't support all params
- Testing code at `/test_api_server.py`


**Finally, please notes that their models are released under the tencent-hunyuan-community license, so I think it's more safe to assume these are noncommercial models.**

While thinking about this model, I was also thinking about test driving the neocloud serverless GPU platforms, and I think 3D generation model is a good "hard test case" for setup.

### Part 2 - Current Attempt

(Actually, a supposedly working `Dockerfile` is also provided in the repo at `/docker/Dockerfile`... so why do we redo this? Maybe because I want to get a more minimalistic image...)

I'm using beam cloud, but the basic principle should be similar on comparable platforms.

Following is the raw source code (very raw due to the expensive trial and error cycle - most of the efforts is concentrated on making the image build at all, and when the image is very complicated like this example, the editing is hard because it's non-interactive and more like coding competition, you scratch your head over why there's this cryptic error message, make a timid attempt to fix it by judiciously changing a single line, then resubmit, wait for 10 minutes for the image build to get back to the point of failure, then hope for the best, and then the "coding competition auto-judge" send you the verdicts. Now to be clear, there are instances in the process where there is opportunity for "trial and error optimization" where some clever rearrangement allows testing more than one hypothesis in a single attempt, and I do try to seize the chance if I can spot it (which however contributes to code redundancy as seen below). As for why it's like this? One reason is that the there is some dependency that have to be compiled directly, and involves the GPU compiler actually (nvcc?), so GPU is necessary during the image build phase also.

```py
from beam import Image, Pod, Volume


VOLUME_PATH = "/workspace/models"
OUTPUT_PATH = "/workspace/outputs"

#magic = "YysrIC1PMyAtV2FsbCAtc2hhcmVkIC1zdGQ9YysrMTEgLWZQSUMgYHB5dGhvbjMgLW0gcHliaW5kMTEgLS1pbmNsdWRlc2AgbWVzaF9pbnBhaW50X3Byb2Nlc3Nvci5jcHAgLW8gbWVzaF9pbnBhaW50X3Byb2Nlc3NvcmBweXRob24zIC1jICJpbXBvcnQgc3lzY29uZmlnOyBwcmludChzeXNjb25maWcuZ2V0X2NvbmZpZ192YXIoJ0VYVF9TVUZGSVgnKSkiYA=="

#magic = "YysrIC1PMyAtV2FsbCAtc2hhcmVkIC1zdGQ9YysrMTEgLWZQSUMgYHB5dGhvbjMgLW0gcHliaW5kMTEgLS1pbmNsdWRlc2AgbWVzaF9pbnBhaW50X3Byb2Nlc3Nvci5jcHAgLW8gbWVzaF9pbnBhaW50X3Byb2Nlc3NvcmBweXRob24zLWNvbmZpZyAtLWV4dGVuc2lvbi1zdWZmaXhg"

magic = "ZWNobyAiSGVsbG8gd29ybGQiCmMrKyAtTzMgLVdhbGwgLXNoYXJlZCAtc3RkPWMrKzExIC1mUElDIGBweXRob24zIC1tIHB5YmluZDExIC0taW5jbHVkZXNgIG1lc2hfaW5wYWludF9wcm9jZXNzb3IuY3BwIC1vIG1lc2hfaW5wYWludF9wcm9jZXNzb3JgcHl0aG9uMy1jb25maWcgLS1leHRlbnNpb24tc3VmZml4YAplY2hvICJCeWUiCg=="

#$(grep -ivE \"^bpy|^#\"

# Create the image with required dependencies
# pytorch image cannot be used because of python version conflict (3.11)
# need build-essential for c++ compiler, coreutils for command line magic
image = (
    Image(
        base_image="nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        #python_version="python3.10"
    )
    .add_commands([
        "apt-get update",
        "apt-get install -y git wget build-essential coreutils",
        "apt install software-properties-common -y",
        "apt-get install libgl1-mesa-dev libxrender1 libfontconfig1 -y", # Lower bound for what's needed - this is not enough
        # TODO: insert an upper bound line that is known to be enough, found from web search
        "add-apt-repository ppa:deadsnakes/ppa -y",
        "apt-get install python3.10 python3.10-dev python3-pip -y"
    ])
    .add_python_packages([
        "torch==2.5.1",
        "torchvision==0.20.1", 
        "torchaudio==2.5.1"
    ])
    # Trailing extra index not allowed by beam
    #.add_python_packages(["bpy==4.1 --extra-index-url https://download.blender.org/pypi/"])
    .add_commands([
        "pip install pybind11==2.13.4",
        "python3 -m pybind11 --includes",
        "python3-config --extension-suffix",
        f"echo {magic} | base64 --decode > /workspace/sample.sh",
        "cat /workspace/sample.sh",
        "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 /workspace/Hunyuan3D-2.1",
        "cd /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && bash /workspace/sample.sh",
        "ls -la /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer",
        "rm /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpython-310-x86_64-linux-gnu.so"
    ])
    .add_commands([
        "cd /workspace/Hunyuan3D-2.1 && pip install bpy==4.0 --extra-index-url https://download.blender.org/pypi/",
        "cd /workspace/Hunyuan3D-2.1 && sed -ie '/bpy/d' requirements.txt",
        "cd /workspace/Hunyuan3D-2.1 && pip install -r requirements.txt",
        "cd /workspace/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && pip install -e .",
        f"echo {magic} | base64 --decode > /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer/compile_mesh_painter_fix.sh",
        "cat /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer/compile_mesh_painter_fix.sh",
        "cd /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter_fix.sh",
        "ls -la /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer",
        "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P /workspace/Hunyuan3D-2.1/hy3dpaint/ckpt"
    ])
    .build_with_gpu(gpu="RTX4090")
)

pod = Pod(
    name="hunyuan3d_2_1_test_gradio",
    ports=[8081],
    cpu=2,
    memory="16Gi",
    gpu="RTX4090",
    image=image,
    volumes=[Volume(name="huggingface_models", mount_path=VOLUME_PATH), Volume(name="output_artifacts", mount_path=OUTPUT_PATH)],
    keep_warm_seconds=300,  # 5 minutes idle timeout,
    env={ "PYTHONPATH": "/workspace/Hunyuan3D-2.1/:/workspace/Hunyuan3D-2.1/hy3dshape/:/workspace/Hunyuan3D-2.1/hy3dpaint/" },
    entrypoint=[
        "python3", "/workspace/Hunyuan3D-2.1/api_server.py", "--model_path", f"{VOLUME_PATH}/tencent/Hunyuan3D-2.1/", "--enable_flashvdm", "--low_vram_mode", "--cache-path", f"{OUTPUT_PATH}/hunyuan3d/"
    ]
)

```

### Part 3 - Explanation of key tricky parts

Let's work through it step by step:

1. Base image choice and implication

`nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` is needed (from the version pin provided above). We cannot use official pytorch image as the comments say, because those image use python 3.11 instead of what we need (python 3.10).

The nvidia image doesn't seem to have python installed? (Or maybe it's just path issue), so we install it again, but python 3.10 may need the addition of the apt repo source. Hence the lines (Notice that pip need to be installed also because `python3.10` by itself doesn't include pip):

```
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt-get install python3.10 python3.10-dev python3-pip -y
```

Though from log it seems to be not necessary - perhaps installation of other apt dependency implicitly installed python 3.10 already.

2. Other system level dependencies

The basics:

```
apt-get install -y git wget build-essential coreutils
```

Again, as comment said: need build-essential for c++ compiler, coreutils for command line magic.

The major tricky part is that for utilities involving processing 3D object, some python libraries listed as dependency in `requirements.txt` implicitly have system level dependencies. For example, they use the OpenGL ecosystem to connect to GPU instead of cuda. The following is from trial and error:

```
"apt-get install libgl1-mesa-dev libxrender1 libfontconfig1 -y"
```

I found from testing that `pymeshlab` needs `libgl1-mesa-dev` due to `libGL.so.1`, while `bpy` (Blender binding) needs `libxrender1` due to `libXi.so.6`, though this is by far not exhaustive. An upper bound list of system dependency comes from peeking at the official Dockerfile.

(TODO: testing by bisection to try to find the minimal dependencies? Or use the list provided at https://github.com/princeton-vl/infinigen/issues/28 instead.)

3. Other gotchas

Then, we can install version pinned pytorch per official instruction.

The official instruction involves running a bash script, that will use C++ compiler to build a single object file. Looking at that script located at `/hy3dpaint/DifferentiableRenderer/compile_mesh_painter.sh`:

```
c++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` mesh_inpaint_processor.cpp -o mesh_inpaint_processor`python3-config --extension-suffix`
```

There is a trap here. In the particular docker setup we used, seems `python` does not link to the python interpreter successfully, while `python3` works. So the subshell execution for `python -m pybind11 --includes` will fail and we need to replaced it. The `python3-config --extension-suffix` required installing the `python3.10-dev` system dependency to work. Although one GitHub issue proposed using vanilla python code in case that util is not available: `python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"`, I couldn't get it to work in the setup above. Not sure if it's due to the multi-nesting-level string escaping in the code above becoming too hard for me to get right, or it really doesn't work. (The multi-nesting-level string escape stuff is what motivated me to use the dark magic of base64 encoding to inject small text file into the image. Actually, it is a common weakness of many of these complex image build situation, where if you need to manually make small code change to the repo, it's hard to do it by command line only, unless you make a diff file and then use `patch` command line tool etc)

Notice that the compilation is silent and have no output, so for observability's sake, we added `echo` line both before *and* after that line in the modified bash script we ended up using.

(Other minor obvious thing: need to be in the correct directory, also need to install `pybind11` first if want to test the bash script)

The other major gotcha, and this is noticed by many as seem in GitHub issue, is that the blender python binding package has an official policy to retire versions outside of the supported range from pypi. Archived versions are still available on their own mirror. Hence need to use `pip install bpy==4.0 --extra-index-url https://download.blender.org/pypi/`, and then remove the `bpy` line from `requirements.txt`.

After that, the rest can go through normally. For reference below is the original installation steps provided officially:

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

cd hy3dpaint/custom_rasterizer
pip install -e .
cd ../..
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..

wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
```

4. The runtime

A final gotcha is that there is relative import problem, especially if you cannot control the image to have the correct current directory `cwd` when running the main process. The official repo seems to know this too and that's why it has the line:

```py
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
```

However, this can be brittle. In our case, solution is to add additional search path python uses by setting the environment variable:

```
export PYTHONPATH="/workspace/Hunyuan3D-2.1/:/workspace/Hunyuan3D-2.1/hy3dshape/:/workspace/Hunyuan3D-2.1/hy3dpaint/"
```


### Part 4 - Testing

Extracted from `/test_api_server.py`:

```py
import requests
import base64
import json
from PIL import Image
import io
import time
import os
image = Image.open("eg1.png").convert("RGBA")
buffer = io.BytesIO()
image.save(buffer, format='PNG')
img_base64 = base64.b64encode(buffer.getvalue()).decode()

result = requests.post(f"{BASE_URL}/generate", json={ "image": img_base64, "type": "glb" }, headers={ "Authorization": f"Bearer {API_KEY}" })
```

Replace `eg1.png` with your image, and supply `BASE_URL` and `API_KEY`.

Be careful that on Window, Ctrl-C won't work (due to system level socket stuff):

https://github.com/psf/requests/issues/5822

Requests ignores KeyboardInterrupt #5822


