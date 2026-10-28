"""PixArt-Sigma Gradio UI -- text-to-image, standalone.

Same shape as the other UIs in this family (Kandinsky, Sana,
Z-Image, SD 3.5, SD 1.5, SDXL): no Jupyter, no Colab. Run it, a
page opens in your browser, type a prompt, pictures appear.

How to run (on a Windows PC with an NVIDIA GPU):

    install.bat      (once -- Python is bundled in python_embeded;
                      it installs torch + the packages in
                      requirements.txt)
    run.bat          (starts the app)

The bundle is FLAT: app.py sits next to run.bat, so pictures land
in images_out/ right here -- one subfolder per batch name, reused
across runs, frames numbered continuously, and settings.txt
appended once per run.

The model is PixArt-alpha/PixArt-Sigma-XL-2-1024-MS on Hugging
Face: public, NOT gated (OpenRAIL++ license) -- no account, no
token, no license form. First Generate downloads it (about
22 GB on disk -- the T5-XXL text encoder ships as float32
weights) and loads it once; repeat generations reuse it.

How it works (Diffusers 0.39.0 -- PixArtSigmaPipeline, whose
call arguments here match its 0.39.0 signature):
  1. A T5-XXL text encoder turns the prompt into embeddings
     (300-token budget, far past CLIP's 77 -- long prompts are
     welcome; the tokenizer is the T5 slow tokenizer, which is
     why the sentencepiece package is in requirements.txt).
  2. A 0.6B diffusion transformer (28 layers, DPM-Solver++
     scheduler as shipped) denoises a latent -- default 20
     steps, guidance 4.5, the pipeline's own defaults.
  3. The VAE (the standard SD VAE, scaling factor 0.13025)
     decodes the latent into the picture.

Memory design -- read this before touching the loader:
  Everything loads in float16, following the model card's own
  usage snippet: T5-XXL 9.5 GB, transformer 1.2 GB, VAE
  0.2 GB -- about 10.9 GB of weights. On a 12 GB card those
  weights plus sampling activations do NOT fit all at once,
  so the pipeline runs with model CPU offload ON by default:
  one stage at a time sits on the GPU (the pipeline's own
  offload order is text_encoder -> transformer -> vae) and
  the rest park in system RAM. Peak RAM is the parked weights
  (worst case the 9.5 GB text encoder) plus the app, which a
  16 GB machine holds; peak VRAM is the text encoder alone
  during encoding, or the small transformer + VAE during
  sampling. The "CPU offload" checkbox can be unticked to put
  everything on the GPU at once -- on a 12 GB card that is
  expected to run out of memory; it is there for bigger cards.

  Batch rule (Oct 28, 2026): a multi-image run encodes its
  prompt ONCE, before the loop, and every image in the run
  is then generated from those saved embeddings -- the text
  encoder never runs inside the batch loop. Reason: with
  CPU offload on, each full pipeline call sweeps the 9.5 GB
  T5-XXL encoder from RAM to the GPU and back, and a
  10-image portrait batch doing that per image exhausted
  Windows commit memory on a 16 GB machine -- python.exe
  died outright (error 0xc000012d, so hard that even the
  Windows error reporter could not start). With the encode
  done once, only the small transformer and VAE cycle per
  image. Single-image runs keep the plain one-call path:
  they never had the problem, so they do not change.

  Dtype rule (the SDXL lesson, Oct 27 2026): ONE dtype story.
  Every component loads in float16 and none is ever upcast
  against the others. Diffusers 0.39's PixArt decode line
  casts latents to the VAE's dtype itself, but the app does
  not rely on reconciliation -- matching dtypes end to end
  is the design, and the stub tests pin it.

  clean_caption is passed as False: the pipeline's caption
  cleaning is meant for web-scraped training captions and
  needs optional packages (ftfy); hand-typed prompts are
  encoded exactly as written.

Sizes: PixArt-Sigma was trained on aspect-ratio buckets. The
presets below are the model's own 1024-family buckets; other
width/height values are snapped by the pipeline to the
nearest bucket internally and the result is resized back, so
the saved picture always matches the width/height boxes.

Not in this version (see the README's deferred list):
img2img, inpainting, ControlNet, LoRAs, embeddings, prompt
weighting/blending, animation. Diffusers ships no standard
PixArt img2img pipeline, so there is nothing honest to wire
those tabs to yet.
"""

import gc
import os
import random
import re
import shutil
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = APP_DIR  # flat bundle: app.py sits next to run.bat
IMAGES_DIR = os.path.join(ROOT_DIR, "images_out")
FAVORITES_DIR = os.path.join(IMAGES_DIR, "favorites")
RANDOM_PROMPTS_FILE = os.path.join(ROOT_DIR, "random_prompts.txt")

MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

DEFAULT_PROMPT = (
    "A cozy mountain cabin with smoke rising from the chimney, "
    "pine forest around it, soft morning light"
)
# PixArt's classifier-free guidance was trained against the
# empty string, so the honest default negative is empty (the
# pipeline's own default too) -- not the SD family's
# "low quality, bad quality". Type one if you want one.
DEFAULT_NEGATIVE = ""

# Samplers offered in the dropdown. All are built from the
# checkpoint's own scheduler config (saved pristine at load),
# so betas and timestep spacing always match the model's
# training. "DPM++ 2M" with no further arguments IS the
# scheduler exactly as the checkpoint ships it.
SAMPLER_NAMES = [
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ SDE",
    "Euler",
    "DDIM",
]
DEFAULT_SAMPLER = "DPM++ 2M"

# The model's own 1024-family aspect buckets (width, height).
_SIZE_PRESETS = {
    "Square 1024x1024": (1024, 1024),
    "Landscape 1152x896": (1152, 896),
    "Portrait 896x1152": (896, 1152),
    "Landscape 1216x832": (1216, 832),
    "Portrait 832x1216": (832, 1216),
    "Wide 1344x768": (1344, 768),
    "Tall 768x1344": (768, 1344),
    "Cinema 1536x640": (1536, 640),
    "Vertical 640x1536": (640, 1536),
}

_pipe = None            # loaded once on first Generate
_pipe_key = None        # the offload flag the loaded pipe used
_scheduler_config = None  # pristine copy of the shipped config


# ------------------------------------------------------------------
# Batch folders, frame numbering, settings.txt -- the house
# helpers, same behaviour as the other UIs.
# ------------------------------------------------------------------

def _safe_batch_name(name):
    """Strip characters Windows forbids in folder names. Falls
    back to a timestamped name when the box is empty."""
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")
    if not name:
        name = datetime.now().strftime("pixart-%Y%m%d-%H%M%S")
    return name


def _batch_folder(base):
    """The one folder for a batch name: images_out/<base>,
    created if missing, reused if it already exists. Repeat
    runs never get a _2/_3 folder -- frames just keep
    numbering upward."""
    return os.path.join(IMAGES_DIR, base)


def _unique_folder(base, root=IMAGES_DIR):
    """<root>/<base>, or <base>_2, <base>_3... if it already
    exists. Used by Star batch, so a starred copy never
    overwrites an earlier one."""
    folder = os.path.join(root, base)
    if not os.path.exists(folder):
        return folder
    i = 2
    while True:
        candidate = "{}_{}".format(folder, i)
        if not os.path.exists(candidate):
            return candidate
        i += 1


_FRAME_RE = re.compile(r"^(\d{6})\.png$")


def _next_frame_number(folder):
    """First unused frame number in a folder: one past the
    highest existing 000001-style frame, or 1 for a fresh
    folder."""
    highest = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            match = _FRAME_RE.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _frame_range_text(start, count):
    """'frame 000006' for one image, 'frames 000006-000008'
    for several -- used in status lines and settings
    headers."""
    if count == 1:
        return "frame {:06d}".format(start)
    return "frames {:06d}-{:06d}".format(start, start + count - 1)


def _seeds_for(seed, count):
    """Resolve the seed box to one seed per image. -1 (or
    empty) draws a fresh random seed per image; a fixed seed
    S gives S, S+1, S+2... so a multi-image run with a fixed
    seed stays reproducible."""
    if seed is None:
        seed = -1
    seed = int(seed)
    if seed < 0:
        return [random.randint(0, 2**31 - 1) for _ in range(count)]
    return [seed + i for i in range(count)]


# Dynamic prompts: single level only, one random pick per
# group. "{a|b|c}" becomes a, b, or c. Groups do not nest; a
# group with no pipe is just its own text with the braces
# removed.
_TEMPLATE_RE = re.compile(r"\{([^{}]*)\}")


def _expand_templates(prompt):
    """Expand {a|b|c} groups, one random pick each. Plain
    prompts (no braces) come back unchanged."""
    return _TEMPLATE_RE.sub(
        lambda match: random.choice(match.group(1).split("|")), prompt
    )


def _append_settings(folder, header, entries):
    """Append one run's block to settings.txt: a header line
    (date, time, frame range) followed by the run's
    'key: value' lines and a blank line. Creates the file on
    first use; earlier runs' blocks are never touched. A
    missing trailing newline in an old file is repaired so
    blocks never run together."""
    path = os.path.join(folder, "settings.txt")
    prefix = ""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(prefix + header + "\n")
        handle.write("\n".join(entries) + "\n\n")


def _run_header(start, count):
    """The settings.txt header for one run: date, time, and
    the frame-number range it produced."""
    return "Run {}, {}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _frame_range_text(start, count))


def _seed_entry(seeds):
    """The settings.txt seed line: 'seed: X' for one image,
    'seeds: X, Y' for several."""
    if len(seeds) == 1:
        return "seed: {}".format(seeds[0])
    return "seeds: {}".format(", ".join(str(s) for s in seeds))


def _expanded_entries(expanded_list):
    """The settings.txt expanded-prompt line(s): one line for
    one image, a comma-joined list for several."""
    if len(expanded_list) == 1:
        return ["expanded_prompt: {}".format(expanded_list[0])]
    return ["expanded_prompts: {}".format(", ".join(expanded_list))]


def _free_memory():
    """Drop cached GPU memory. Best effort: never raises, so a
    cleanup step cannot break the run that called it. Called
    after every generated image, so a long queue does not
    creep upward in memory use."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- freeing memory must not fail
        pass


# ------------------------------------------------------------------
# History -- last 10 jobs, most recent first. Each entry keeps
# the job's settings so picking it restores them, not just the
# prompt text.
# ------------------------------------------------------------------

def _history_add(history, entry):
    """Prepend a job entry (a dict) to the session history:
    most recent first, last 10, duplicates by prompt
    collapsed."""
    history = [e for e in (history or [])
               if e.get("prompt") != entry.get("prompt")]
    history.insert(0, entry)
    return history[:10]


def _history_update(history):
    """The history dropdown update: choices are the entry
    prompts, most recent first, value cleared so picking the
    same prompt twice still fires."""
    import gradio as gr
    choices = [e.get("prompt", "") for e in (history or [])]
    return gr.update(choices=choices, value=None)


def _history_nochange():
    """A dropdown update that changes nothing (early exits)."""
    import gradio as gr
    return gr.update()


def _history_restore(choice, history):
    """Return the control values stored for the picked history
    entry: prompt, negative, width, height, steps, guidance,
    sampler, seed. Falls back to the prompt text alone (with
    the app's defaults for the rest) when the entry is not
    found."""
    import gradio as gr
    for entry in (history or []):
        if entry.get("prompt") == choice:
            return (
                entry.get("prompt", ""),
                entry.get("negative", DEFAULT_NEGATIVE),
                entry.get("width", 1024),
                entry.get("height", 1024),
                entry.get("steps", 20),
                entry.get("guidance", 4.5),
                entry.get("sampler", DEFAULT_SAMPLER),
                entry.get("seed", -1),
            )
    return (choice or "", DEFAULT_NEGATIVE, 1024, 1024, 20, 4.5,
            DEFAULT_SAMPLER, -1)


def _early(message, last_folder, history):
    """The generate() return tuple for an early exit: empty
    gallery, the message, no history change."""
    return [], message, _history_nochange(), last_folder, history


# ------------------------------------------------------------------
# Model loading -- one pipeline, float16 throughout, model CPU
# offload by default (see the header for the memory maths).
# ------------------------------------------------------------------

def _unload_all():
    """Drop the loaded pipeline and the saved scheduler
    config, and free the memory (VRAM and the RAM the
    offloaded weights were parked in)."""
    global _pipe, _pipe_key, _scheduler_config
    _pipe = None
    _pipe_key = None
    _scheduler_config = None
    _free_memory()


def unload_models():
    """Free the generator (VRAM and RAM)."""
    _unload_all()
    return "Models unloaded -- VRAM and RAM freed."


def _load_pipe(offload):
    """Load (or reuse) the PixArt-Sigma pipeline.

    Everything loads in float16 -- the model card's own usage
    snippet -- and NO component is moved to a different dtype
    afterwards: one dtype story end to end (the SDXL lesson).
    With offload on (the default), enable_model_cpu_offload
    parks the components in system RAM and brings one stage
    at a time onto the GPU, in the pipeline's declared order
    (text_encoder -> transformer -> vae). With offload off,
    the whole 10.9 GB sits on the GPU at once.

    The pristine scheduler config is saved at load so sampler
    swaps always start from the checkpoint's own config.

    Raises RuntimeError with a plain sentence on failure."""
    global _pipe, _pipe_key, _scheduler_config
    key = bool(offload)
    if _pipe is not None and _pipe_key == key:
        return _pipe
    _unload_all()

    import torch
    from diffusers import PixArtSigmaPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. PixArt-Sigma UI needs an NVIDIA "
            "GPU with a recent driver -- CPU-only mode is not "
            "supported.")
    try:
        pipe = PixArtSigmaPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash
        raise RuntimeError(
            "Could not load {} ({}). Check the internet "
            "connection and that Hugging Face is reachable, "
            "then try Generate again.".format(MODEL_ID, exc))
    if key:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    _pipe = pipe
    _pipe_key = key
    _scheduler_config = dict(pipe.scheduler.config)
    return pipe


def _loader_text(offload):
    """How the model is placed, in the words settings.txt and
    the status line use -- so a PC test can confirm the
    intended loader path actually ran."""
    if offload:
        return "model cpu offload"
    return "full gpu (offload off)"


def _make_scheduler(name):
    """Build the scheduler for a sampler name from the saved
    pristine config (so betas and timestep spacing always
    match the checkpoint's training).

    The checkpoint ships DPMSolverMultistepScheduler
    (algorithm dpmsolver++, midpoint, order 2, Karras sigmas
    OFF). Mappings:
      DPM++ 2M         -> that scheduler, config untouched
      DPM++ 2M Karras  -> same, with use_karras_sigmas=True
      DPM++ SDE        -> DPMSolverSinglestepScheduler, karras
      Euler            -> EulerDiscreteScheduler
      DDIM             -> DDIMScheduler
    An unknown name falls back to the native scheduler."""
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerDiscreteScheduler,
    )
    config = _scheduler_config
    if name == "DPM++ 2M Karras":
        return DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True)
    if name == "DPM++ SDE":
        return DPMSolverSinglestepScheduler.from_config(
            config, use_karras_sigmas=True)
    if name == "Euler":
        return EulerDiscreteScheduler.from_config(config)
    if name == "DDIM":
        return DDIMScheduler.from_config(config)
    return DPMSolverMultistepScheduler.from_config(config)


def _make_generator(seed):
    """A CPU generator for one image (the house pattern:
    Diffusers moves it to the right device itself, and a CPU
    generator is what model CPU offload expects)."""
    import torch
    return torch.Generator("cpu").manual_seed(int(seed))


# ------------------------------------------------------------------
# Random prompt -- random_prompts.txt next to app.py, read
# fresh on every click; the built-in word banks are the
# fallback when the file is missing or empty.
# ------------------------------------------------------------------

_SUBJECTS = [
    "a lighthouse", "a fox", "a mountain cabin", "a castle",
    "a harbor town", "a red panda", "a sailing ship",
    "a robot gardener", "a treehouse village", "a steam train",
]
_SETTINGS = [
    "a rocky cliff", "a misty forest", "desert dunes",
    "a snowy village", "a neon city street", "a quiet lake",
    "a sunflower field", "a space station", "a coral reef",
    "an old library",
]
_LIGHTS = [
    "golden hour light", "storm light", "moonlight",
    "neon dusk", "soft morning fog", "candlelight",
    "sunset clouds", "starlight",
]
_STYLES = [
    "photorealistic", "oil painting", "anime style",
    "watercolor", "cyberpunk", "storybook illustration",
]


def random_prompt():
    """One random prompt: a line from random_prompts.txt when
    the file has lines, else the built-in word banks."""
    try:
        with open(RANDOM_PROMPTS_FILE, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        if lines:
            return random.choice(lines)
    except OSError:
        pass
    return "A {} in {}, {}, {}, detailed.".format(
        random.choice(_SUBJECTS),
        random.choice(_SETTINGS),
        random.choice(_LIGHTS),
        random.choice(_STYLES),
    )


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------

def _run_generate(pipe, template, negative, width, height, steps,
                  guidance, seeds, folder, start=1):
    """One text-to-image generation per seed, each with its
    own generator, so each image's seed reproduces it. Frames
    are numbered from start (6-digit), so a repeat run in a
    reused folder continues where the last run left off.
    Memory is freed after every image (the walk-away rule: a
    long queue must not creep upward). Returns (frame paths,
    expanded prompts actually used).

    Two paths, by seed count. A single-seed run takes the
    plain path: one pipeline call with the prompt text,
    expanded fresh -- the proven first-picture path,
    unchanged. A multi-seed run encodes once (the module
    header's batch rule): the template is expanded ONE time,
    pipe.encode_prompt computes the embeddings before the
    loop, and each seed is generated from those same
    embeddings with prompt=None, so the 9.5 GB text encoder
    never fires inside the loop and the CPU-offload hooks
    never sweep it per image. The returned expanded list
    repeats that one expansion once per frame, which is what
    settings.txt records."""
    frame_paths = []
    expanded_list = []
    if len(seeds) > 1:
        expanded = _expand_templates(template)
        # encode_prompt returns, in the diffusers 0.39.0
        # wheel's own order: prompt_embeds,
        # prompt_attention_mask, negative_prompt_embeds,
        # negative_prompt_attention_mask (the negative pair
        # is None when guidance is off).
        (prompt_embeds, prompt_attention_mask,
         negative_prompt_embeds,
         negative_prompt_attention_mask) = pipe.encode_prompt(
            expanded,
            do_classifier_free_guidance=float(guidance) > 1.0,
            negative_prompt=negative,
            num_images_per_prompt=1,
            clean_caption=False)
        for i, actual_seed in enumerate(seeds):
            generator = _make_generator(actual_seed)
            image = pipe(
                prompt=None,
                # negative_prompt must be None here, not the
                # pipeline's "" default: check_inputs refuses
                # a text negative forwarded alongside negative
                # prompt embeds (Oct 28 2026 PC test).
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=(
                    negative_prompt_attention_mask),
                height=int(height),
                width=int(width),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator,
                clean_caption=False,
            ).images[0]
            frame_path = os.path.join(
                folder, "{:06d}.png".format(start + i))
            image.save(frame_path)
            frame_paths.append(frame_path)
            expanded_list.append(expanded)
            _free_memory()
        return frame_paths, expanded_list
    for i, actual_seed in enumerate(seeds):
        expanded = _expand_templates(template)
        generator = _make_generator(actual_seed)
        image = pipe(
            prompt=expanded,
            negative_prompt=negative,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
            clean_caption=False,
        ).images[0]
        frame_path = os.path.join(folder, "{:06d}.png".format(start + i))
        image.save(frame_path)
        frame_paths.append(frame_path)
        expanded_list.append(expanded)
        _free_memory()
    return frame_paths, expanded_list


def _generate_entries(template, expanded_list, negative, width,
                      height, steps, guidance, sampler, seeds,
                      offload):
    """settings.txt entries for a text-to-image run. The
    expanded prompt line is what was actually used (the
    template line keeps what was typed). The loader line
    records which placement path ran."""
    return (
        ["prompt: {}".format(template)]
        + _expanded_entries(expanded_list)
        + ["negative_prompt: {}".format(negative or ""),
           "sampler: {}".format(sampler),
           "width: {}".format(int(width)),
           "height: {}".format(int(height)),
           "inference_steps: {}".format(int(steps)),
           "guidance_scale: {}".format(float(guidance))]
        + [_seed_entry(seeds),
           "model: {}".format(MODEL_ID),
           "dtype: float16",
           "loader: {}".format(_loader_text(offload)),
           "device: cuda"]
    )


def generate(prompt_text, negative_text, queue_text, width, height,
             steps, guidance, sampler, seed, batch_name, count,
             offload, last_folder, history):
    """Text-to-image generation. Returns (gallery image list,
    status text, history dropdown update, last batch folder,
    history list). The gallery shows the saved frames.

    If the prompt queue has any non-empty lines, the queue
    runs INSTEAD of the single prompt box: each line is its
    own run with the current settings, sharing the one batch
    folder, frames numbered continuously across lines, one
    settings block per line. Otherwise the single prompt box
    runs. Repeat runs with the same batch name reuse the
    folder and keep numbering upward, so nothing is ever
    overwritten."""
    import torch

    lines = [line.strip() for line in (queue_text or "").splitlines()
             if line.strip()]
    if not lines and (not prompt_text or not prompt_text.strip()):
        return _early("Type a prompt first.", last_folder, history)
    if not torch.cuda.is_available():
        return _early(
            "No CUDA GPU found. PixArt-Sigma UI needs an NVIDIA "
            "GPU with a recent driver.", last_folder, history)
    try:
        pipe = _load_pipe(offload)
    except Exception as exc:  # noqa: BLE001 -- show it, don't crash
        return _early("Model load failed: {}".format(exc),
                      last_folder, history)
    pipe.scheduler = _make_scheduler(sampler)

    # Images per Generate is clamped to 1-10: the box already
    # limits it, but a typed value can slip past, so the code
    # enforces it.
    count = int(count) if count else 1
    count = max(1, min(10, count))
    negative = (negative_text or "").strip()
    history = history or []

    folder = _batch_folder(_safe_batch_name(batch_name))
    os.makedirs(folder, exist_ok=True)

    def run_one(template):
        """One prompt (box or queue line) at the current
        settings; returns (frames, seeds)."""
        seeds = _seeds_for(seed, count)
        start = _next_frame_number(folder)
        frames, expanded = _run_generate(
            pipe, template, negative, width, height, steps,
            guidance, seeds, folder, start)
        _append_settings(
            folder, _run_header(start, len(frames)),
            _generate_entries(template, expanded, negative, width,
                              height, steps, guidance, sampler,
                              seeds, offload))
        entry = {
            "prompt": template,
            "negative": negative,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "guidance": float(guidance),
            "sampler": sampler,
            "seed": int(seed) if seed is not None else -1,
        }
        return frames, seeds, start, entry

    if lines:
        all_frames = []
        all_seeds = []
        first_frame = None
        for line in lines:
            frames, seeds, start, entry = run_one(line)
            if first_frame is None:
                first_frame = start
            all_frames.extend(frames)
            all_seeds.extend(seeds)
            history = _history_add(history, entry)
        status = (
            "Queue done: {} prompts, {} images. Saved to {} "
            "({}, {}). Loader: {}.".format(
                len(lines), len(all_frames), folder,
                _frame_range_text(first_frame, len(all_frames)),
                _seed_entry(all_seeds), _loader_text(offload)))
        return (all_frames, status, _history_update(history),
                folder, history)

    frames, seeds, start, entry = run_one(prompt_text)
    history = _history_add(history, entry)
    status = "Saved to {} ({}, {}). Loader: {}.".format(
        folder, _frame_range_text(start, len(frames)),
        _seed_entry(seeds), _loader_text(offload))
    return (frames, status, _history_update(history), folder,
            history)


def star_batch(last_folder):
    """Copy the most recently generated batch folder into
    images_out/favorites/ under a unique name (never
    overwrite). If nothing has run yet this session, say so
    in the status."""
    if not last_folder or not os.path.isdir(last_folder):
        return "No batch has run yet in this session."
    os.makedirs(FAVORITES_DIR, exist_ok=True)
    base = os.path.basename(os.path.normpath(last_folder))
    dest = _unique_folder(base, root=FAVORITES_DIR)
    shutil.copytree(last_folder, dest)
    return "Starred to {}".format(dest)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

def _apply_preset(preset):
    """Set the width/height sliders from a size preset name."""
    width, height = _SIZE_PRESETS.get(preset, (1024, 1024))
    return width, height


def build_ui():
    import gradio as gr

    with gr.Blocks(title="PixArt-Sigma UI") as demo:
        gr.Markdown(
            "# PixArt-Sigma UI\n"
            "Text-to-image on your own GPU -- no Jupyter, no "
            "Colab. First Generate downloads the model (about "
            "22 GB) and loads it with CPU offload, so it takes "
            "a while; after that, repeat runs are just the "
            "render."
        )
        # Session-only state: never written to disk.
        last_folder_state = gr.State(None)
        history_state = gr.State([])

        prompt = gr.Textbox(label="Prompt", lines=3,
                            value=DEFAULT_PROMPT)
        negative_prompt = gr.Textbox(
            label="Negative prompt (PixArt's native default is "
                  "empty)", lines=2, value=DEFAULT_NEGATIVE)
        queue_text = gr.Textbox(
            label="Prompt queue (one prompt per line)", lines=4)
        gr.Markdown(
            "When the queue has lines, Generate runs the queue "
            "instead of the prompt above -- one run per line, "
            "all sharing the batch folder, frames numbered "
            "continuously, one settings block per line."
        )
        with gr.Row():
            random_btn = gr.Button("Random prompt")
        with gr.Row():
            preset = gr.Dropdown(
                label="Size preset (the model's aspect buckets)",
                choices=list(_SIZE_PRESETS),
                value="Square 1024x1024")
        with gr.Row():
            width = gr.Slider(512, 1536, step=64, value=1024,
                              label="Width")
            height = gr.Slider(512, 1536, step=64, value=1024,
                               label="Height")
        gr.Markdown(
            "Sizes off the presets are snapped by the model to "
            "its nearest trained bucket and resized back, so "
            "the saved picture always matches these boxes."
        )
        with gr.Row():
            sampler = gr.Dropdown(label="Sampler",
                                  choices=SAMPLER_NAMES,
                                  value=DEFAULT_SAMPLER)
            steps = gr.Slider(1, 50, step=1, value=20,
                              label="Inference steps")
            guidance = gr.Slider(1.0, 10.0, step=0.5, value=4.5,
                                 label="Guidance scale")
        with gr.Row():
            seed = gr.Number(value=-1, precision=0,
                             label="Seed (-1 = random)")
            count = gr.Number(value=1, precision=0, minimum=1,
                              maximum=10,
                              label="Images per Generate")
        offload = gr.Checkbox(
            value=True,
            label="CPU offload (recommended: the T5 text "
                  "encoder alone is 9.5 GB -- off keeps every "
                  "stage but one in RAM; unticking puts all "
                  "10.9 GB on the GPU and can run out of memory "
                  "on a 12 GB card)")
        batch_name = gr.Textbox(label="Batch name",
                                value="pixart-test")
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            stop_btn = gr.Button("Stop")
            star_btn = gr.Button("Star batch")
            unload_btn = gr.Button("Unload models")
        gallery = gr.Gallery(label="Results")
        status = gr.Textbox(label="Status")
        history = gr.Dropdown(
            label="History (last 10 -- picking one restores its "
                  "prompt and settings)",
            choices=[], interactive=True)

        random_btn.click(random_prompt, outputs=[prompt])
        preset.change(_apply_preset, inputs=[preset],
                      outputs=[width, height])
        generate_event = generate_btn.click(
            generate,
            inputs=[prompt, negative_prompt, queue_text, width,
                    height, steps, guidance, sampler, seed,
                    batch_name, count, offload, last_folder_state,
                    history_state],
            outputs=[gallery, status, history, last_folder_state,
                     history_state],
        )
        # Gradio 5 cancel pattern: Stop cancels the running
        # Generate event (same as the other UIs).
        stop_btn.click(fn=None, inputs=[], outputs=[],
                       cancels=[generate_event])
        star_btn.click(star_batch, inputs=[last_folder_state],
                       outputs=[status])
        unload_btn.click(lambda: unload_models(), outputs=[status])
        history.change(
            _history_restore,
            inputs=[history, history_state],
            outputs=[prompt, negative_prompt, width, height,
                     steps, guidance, sampler, seed],
        )
    return demo


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    build_ui().launch(server_name="127.0.0.1", inbrowser=True)
