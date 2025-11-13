# Ming Lite Omni v1.5

## Info card - key links

- Quanted model page: https://huggingface.co/wikeeyang/Ming-Lite-Omni-v1.5-NF4
- Original/Official model page: https://huggingface.co/inclusionAI/Ming-Lite-Omni-1.5
- Inference code repo (combined for v1, v1.5, and newer versions): https://github.com/inclusionAI/Ming

## Background

> Model page (for reference only, we'll also use other pages in practise): https://huggingface.co/inclusionAI/Ming-Lite-Omni-1.5

**Natively Any-to-any multimodal LLM** is one of the important goal when trying to extend LLM's capability. Two reasons for the goal's significance:

1. **Any-to-any** implies that the model is flexible enough to process data in any media/modalities, and many tasks traditionally classified by different name can be subsumed under the same umbrella (eg Text-to-speech (TTS) is `text (input) -> audio (output)`, OCR is `image (input) -> text (output)`, image editing is `text + image (input) -> image (output)`, and so on and so on). Here, it is crucial that flexibility exists on both the input and output side.
2. Being **native** points in the direction that the model is trained using early fusion method (i.e. multimodal data already used at massive scale in pretraining phase), instead of freezing a text only model and training an adaptor or full fine-tuning using multimodal dataset. A wish among some is that doing so with sufficiently large scale would lead to the emergence of a rich, multimodal **world model**, where the AI have deep understanding of multiple modalities. One possible benefit is that this increase the chance for the model to be able to generalize to new tasks better by task composition/interpolation, leveraging its knowledge of the interactions between modalities.

Together, it suggests a powerful combo where the model is able to perform many multimodal tasks in a seamless and generalizable way, expanding the range of tasks it can perform compared to a text only LLM.

That being said, it has been a steady but slow ride for open weights model in trying to reach this milestone. Aside from the increase in difficulty of training, and a less mature ecosystem, from the programming side, incorporating multimodal supports in inference engine software have also been a slow, difficult struggle. We've seen input image modality being the first (relatively speaking) to get supports and adoption. Audio input and image output however have lagged further behind, only finally seeing more successful attempt more recently (again, relatively speaking - we need to temper our expectations but keep the hope).

Very recently, there is finally some open weight model release that seem to be getting closer - [Ming Flash Omni (preview) by inclusionAI of Ant group (104B)](https://huggingface.co/inclusionAI/Ming-flash-omni-Preview) and [Emu 3.5 by BAAI (33B)](https://huggingface.co/BAAI/Emu3.5). But these models are quite large and resource heavy to run. At the same time, if there is a trends toward more models of these type being released in the future, then it make sense to practise installing/setting up these models using a smaller size variant as practise rounds.

So, in this document, we share the trial and error troubleshooting process we had when trying to install the model. The model is an earlier and more lightweight version of Ming Omni.

## Installation

### Part 1 - Basic information gathering, planning, and quantization

The model, Ming Lite Omni 1.5, has size 19B param. Even so, we need quantization if we want to run it using a single 24GB VRAM GPU card (it is fortunate that this work at all - for larger model, often even quantization won't be enough and H100 card or multi-GPU, or both, would be necessary).

**Credits to user wikeeyang for providing the quant and sharing test results**

https://github.com/inclusionAI/Ming/issues/42

User reported successfully running a quanted version on a single RTX 4090, and also shared the version pinning to get it working:

- Version pin: Ubuntu 22.04, Python 3.11, Cuda 12.4, Torch 2.6.0.
- VRAM usage: 20 - 23 GB (have reports that disabling the audio feature will shave 2GB off, while disabling the image generation feature shave 6GB off)

There are also other valuable info gleamed from that discussion thread:

- significant code change between v1 and v1.5
- gradio demo breaking on newer version (?)
- When using quant version, the quant model repo only provide the changed file. For multimodal components, would still need to download from the original/official model repo

Because of this, we need to prepare more by tracing down which commit of the repo would actually work. We identified:

https://github.com/inclusionAI/Ming/pull/43

> Gradio demo updated #43 
>
> clzhou commented on Jul 28
>
> Changes include:
> (1) The implementation of gradio demo is updated to be compatible with Ming-Lite-Omni-v1.5.
> (2) The image figures/cases/reasoning.png is replaced with the correct image.

And the relevant git commit is `d97e2f3`.

Also, due to the separate model file mentioned above, downloading model is more complicated - we need to remember to selectively download files from both huggingface model repo and place them in the correct location after download.

Finally, we do still read the official instructions found on the github repo and huggingface model card. But experience shows that usually actual installation is rarely as smooth as just executing the provided instruction.

Actually, one more thing. There is a special reason that, out of the large diversity of not-a-vanilla-LLM model types, we picked this model first. inclusionAI is a slightly more experimental lab compared to the labs releasing *mainstream* models, so it is expected that things would be rougher around the edge. Also, as we will see, older repo are more challenging to reproduce because they are more likely to be stuck on older/legacy version of key supporting software. (Though there are reasons, the main mission of labs are to do AI research, upgrading code to supports newer version of the underlying tech stack may require lots of engineering efforts to migrate and smooth out all the issues/incompatibilities. Unless the model is intended for long term development, it might make more sense to just version pin everything and call it a day and move on)


### Part 2 - Actual installation, preliminary setups before tricky parts

We use a freshly provisioned VM, running Ubuntu.

(Do not execute the command below until you read the notice below)

Download quant model and inference code, version pin the inference code to known-good commit, also install version pinned version of basic pytorch.

```
pip install huggingface_hub
hf download wikeeyang/Ming-Lite-Omni-v1.5-NF4 --local-dir <choose a loc>
git clone https://github.com/inclusionAI/Ming.git
cd Ming
git checkout d97e2f3
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Then, download the multimodal components from original model repo:

```
hf download inclusionAI/Ming-Lite-Omni-1.5 --include "connector/*" --local-dir <choose a loc>
hf download inclusionAI/Ming-Lite-Omni-1.5 --include "mlp/*" --local-dir <choose a loc>
hf download inclusionAI/Ming-Lite-Omni-1.5 --include "scheduler/*" --local-dir <choose a loc>
hf download inclusionAI/Ming-Lite-Omni-1.5 --include "transformer/*" --local-dir <choose a loc>
hf download inclusionAI/Ming-Lite-Omni-1.5 --include "vae/*" --local-dir <choose a loc>
hf download inclusionAI/Ming-Lite-Omni-1.5 --include "talker/*" --local-dir <choose a loc>
```

For the two steps above, **it is recommended to choose a local path you control**. See https://huggingface.co/docs/huggingface_hub/guides/cli#download-to-a-local-folder . The reason is because huggingface uses a symlink design, and since we need to "merge" files from two repo, and neither of them are simple single file case (research oriented HF library `transformer`/`diffuser` models are a whole folder, while GGUF is single file and designed for easy distribution to model user). **If you don't do this, you can still make it work, but you would need some tedious shell scripting.** Example of what I am talking about:

```
ln -s $(readlink -f /teamspace/studios/this_studio/.cache/huggingface/hub/models--wikeeyang--Ming-Lite-Omni-v1.5-NF4/snapshots/56b2c47cb7d14a08b7fcd9f3aa5e17aade8eefae/config.json) /teamspace/studios/this_studio/.cache/huggingface/hub/models--inclusionAI--Ming-Lite-Omni-1.5/snapshots/a1085e8483509daf34024b434591b09205d0a384/config.json
```

*and* repeat for each file, either manually, or via some clever shell script. That, or save yourself the labour by simply doing it right the first time.

### Part 3 - Core trick - flash attention death trap, and other minor trick

First let's get the minor trick out of the way. It's not mentioned in official doc (and not their fault since official doc is usually not responsible for documenting third party quants). Because we are using quants within the huggingface ecosystem, we also need to install `bitsandbytes`:

```
pip install bitsandbytes
```

Now for the main dish so to speak. On first glance, flash attention seems like a simple, straight forward install. After all, it is just a small and nice paper that propose a map-reduce style computational method, and its implementation is just a small custom GPU kernel. Certainly nothing compared to big libraries like pytorch, right? Actually, not so fast. Remember how in C++ there is templating/meta-programming and if you're not careful, just the pre-compilation step of computing the template substutition could take ages? Well that's what happened historically for flash attention. There is no explicit warning about this, so if you don't know, you can accidentally end up having the build/installation take hours. If you are doing this on a pay-as-you-go rented GPU cloud computes... well, sorry about the bill (the GPU is idle during compilation).

Let's investigate and look at some references:

https://github.com/Dao-AILab/flash-attention/issues/1038

Build flash-attn takes a lot of time  #1038


https://github.com/Dao-AILab/flash-attention/issues/1038#issuecomment-2439430999

> Upgrading pip, wheel and setuptools helped me improve the compile time a lot.
>
> python -m pip install --upgrade pip wheel setuptools
>
> Also consider manually setting the number of jobs (64 requires ~500GB ram so adjust accordingly).
>
> MAX_JOBS=64 python -m pip -v install flash-attn --no-build-isolation
>
> Without both changes it defaulted to just using a single compilation job for me, taking forever (I gave up after an hour).
>
> Maybe this could be added to the ninja disclaimer in the readme, @tridao , although I guess the recommended nvidia container has matching versions installed already.


Other similar references:
- https://stackoverflow.com/questions/79525834/how-to-compile-flashattention-wheels-faster
- https://til.simonwillison.net/python/installing-flash-attention


https://gist.github.com/andygrove/c3c517f3b006f7637be171f4e066464f

From reading this, I got the steps seem to be (and checking my bash history this seems to be the solution I ended up using):

```
sudo apt install ninja-build
python -m pip install --upgrade pip wheel setuptools

MAX_JOBS=8 python -m pip -v install flash-attn==2.3.6 --no-build-isolation
```

Also a tip, unlike other programming language, in python, dependency management is stateful and mutative by default - it is not like you give a list, the manager solve for the constraint, then gives you a reproducible build. This can be both an advantage and a disadvantage. So:

http://stackoverflow.com/questions/19548957/ddg#19549035

```
pip install --upgrade --force-reinstall <package>
```

At least, during the trial error phase when you need to fix things/retry after figuring out what corrections are needed.

Now, as an aside: some of you may wonder, if flash attention, which is a key dependency used by numerous AI models, both mainstream and research oriented ones, has such a major trap in it, then why don't we see disaster left and right? Well, because after the disaster, they patched the issue somewhat - in newer version, they provide pre-built wheel files, and if your system is compatible with one of their support matrix combination, pip will just download wheel binary and install directly, no need to compile from source. The reason we got hit here is precisely because we are stuck on older version due to using the version pin technique, where we fell out of the pre-built binary support matrix. **(Disclaimer: even with this trap, I still believe that version pinning, especially on tricky repos like these, is a net positive)**


Anyway, after that, remember to also install `grouped_gemm`:

```
pip install -U grouped_gemm==0.1.4 --no-build-isolation
```

And then, or before (forgot the order), install the main requirement file (after changing to the Ming repo folder):

```
python -m pip install -r requirements.txt
```

Now, edit the source code in the Ming repo, mainly because we are using the quant version instead. Due to the merging necessary, we cannot use the huggingface remote/cloud repo id specifier, and should instead points to a local dir explicitly.

One more code change is necessary:

```
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL pathlib.PosixPath was not an allowed global by default. Please use `torch.serialization.add_safe_globals([PosixPath])` or the `torch.serialization.safe_globals([PosixPath])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
```

Or, without code change, set the environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD. See:

https://huggingface.co/coqui/XTTS-v2/discussions/113

Can you tell me how to fix it, I'm using the example given by the official #113 by zjzjzjzjzjzjzj - opened Apr 16 

(TODO: Check the code change on my VM)

Yes, many research model still use pytorch native serialization with code execution risk, instead of the safetensor format...


And then you can do a basic smoke test on text only inference:

```
python hello.py
```

Yeah!

### Part 4 - Getting multimodal and gradio demo to work, old trick in new skin

I mean, I remember I have faced the exact same bug/issue before, in a completely different scenario.

So I thought, it's not too hard right? Let's just do it:

```
pip install data/matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl
pip install gradio==4.44.1
pip install wget
```

There will be errors. Again, due to version pinning, we need to downgrade gradio. The version that I ended up on:

```
IMPORTANT: You are using gradio version 3.50.2, however version 4.44.1 is available, please upgrade.
```

But also, gradio has a famous version incompatibility issue, references:

- ERROR: Exception in ASGI application gradio-app/gradio#10662 (comment)
- application does not launch gradio-app/gradio#10649 (comment)

https://github.com/XPixelGroup/DiffBIR/issues/176

if "const" in schema: TypeError: argument of type 'bool' is not iterable #176

> Fix by downgrade pydantic.
>
> pip install pydantic==2.10.6
>
> Writing into requirements.txt may be helpful.


When all is said and done, finally, it should work...

```
python gradio_demo.py
```


