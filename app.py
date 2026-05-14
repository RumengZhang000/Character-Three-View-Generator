"""
Character Three-View Generator
A generative AI pipeline for producing character turnaround sheets
with three full-body T-pose views (front, side, back).

Pipeline architecture:
    Stage 1 - SDXL ControlNet generates the base layout using a
              Canny-edge skeleton derived from an anatomical template.
    Stage 2 - YOLOv8 face detector locates faces in the base output.
    Stage 3 - SDXL Inpainting refines each detected face with
              context-aware cropping and a soft elliptical mask.

Dual base-model architecture:
    - Animagine XL 3.1 for 2D anime style
    - DreamShaper XL 1.0 for 3D stylized concept art
"""

import os

# Configure HuggingFace cache directory (must be set before importing diffusers/transformers)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HF_CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_models")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

import gc
import time
import json
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from ultralytics import YOLO
import gradio as gr


# ----------------------------------------------------------------------
# Model configuration
# ----------------------------------------------------------------------
INPAINT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_ID    = "diffusers/controlnet-canny-sdxl-1.0"

MODEL_CONFIGS = {
    "2D Anime": {
        "base": "cagliostrolab/animagine-xl-3.1",
        "use_fp16_variant": True,
        "quality_pos": "masterpiece, best quality, very aesthetic, absurdres",
        "quality_neg": (
            "lowres, bad anatomy, bad hands, text, error, "
            "missing fingers, extra digit, fewer digits, "
            "worst quality, low quality, normal quality, "
            "jpeg artifacts, signature, watermark, username, "
            "blurry, artist name, unfinished"
        ),
    },
    "3D Stylized": {
        "base": "Lykon/dreamshaper-xl-1-0",
        "use_fp16_variant": False,
        "quality_pos": "masterpiece, best quality, highly detailed, professional concept art, sharp focus",
        "quality_neg": (
            "low quality, blurry, bad anatomy, deformed, mutated, "
            "extra limbs, missing limbs, malformed hands, "
            "watermark, text, signature, jpeg artifacts, "
            "amateur photo, snapshot, low quality photo"
        ),
    },
}

# Output canvas: 21:9 ultra-wide, within SDXL's native pixel budget
OUT_W, OUT_H = 1536, 640

# Built-in anatomical templates (located in ./templates/)
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "templates")
BUILTIN_TEMPLATES = {
    "Female (default)": "woman.jpg",
    "Male":             "man.jpg",
    "Child":            "kid.jpg",
    "Custom upload":    None,
}


# ----------------------------------------------------------------------
# Initial model loading (shared across base-model switches)
# ----------------------------------------------------------------------
print("Loading YOLOv8 face detector...")
face_detector = YOLO("yolov8n-face-lindevs.pt")

print("Loading SDXL ControlNet (Canny)...")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=torch.float16)

# Base pipelines are loaded on-demand and cached
_cached_base_pipe = None
_cached_base_id = None
_cached_inpaint_pipe = None


def load_base_pipe(style):
    """Load or retrieve cached base pipeline for the given style."""
    global _cached_base_pipe, _cached_base_id
    config = MODEL_CONFIGS[style]
    target_id = config["base"]

    if _cached_base_id == target_id and _cached_base_pipe is not None:
        return _cached_base_pipe

    # Free previous base model from VRAM before loading new one
    if _cached_base_pipe is not None:
        print(f"Unloading {_cached_base_id}...")
        del _cached_base_pipe
        _cached_base_pipe = None
        _cached_base_id = None
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Loading base model: {target_id}")
    base_kwargs = dict(
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    # Try both fp16 variant and standard loading
    preferred_first = config.get("use_fp16_variant", False)
    attempts = (
        [{"variant": "fp16"}, {}]
        if preferred_first
        else [{}, {"variant": "fp16"}]
    )

    pipe = None
    last_error = None
    for attempt in attempts:
        try:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                target_id, **{**base_kwargs, **attempt}
            )
            break
        except Exception as e:
            last_error = e
            continue

    if pipe is None:
        raise RuntimeError(f"Failed to load {target_id}: {last_error}")

    pipe.enable_model_cpu_offload()
    # Override scheduler config to avoid base-model-specific quirks
    # (DreamShaper XL ships with deis/zero combination that diffusers rejects)
    scheduler_config = dict(pipe.scheduler.config)
    scheduler_config["algorithm_type"] = "dpmsolver++"
    scheduler_config["final_sigmas_type"] = "zero"
    scheduler_config["use_karras_sigmas"] = True
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
    pipe.enable_vae_tiling()

    _cached_base_pipe = pipe
    _cached_base_id = target_id
    return pipe


def load_inpaint_pipe():
    """Load or retrieve cached SDXL inpainting pipeline."""
    global _cached_inpaint_pipe
    if _cached_inpaint_pipe is not None:
        return _cached_inpaint_pipe

    print("Loading SDXL Inpainting pipeline...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.enable_model_cpu_offload()
    scheduler_config = dict(pipe.scheduler.config)
    scheduler_config["algorithm_type"] = "dpmsolver++"
    scheduler_config["final_sigmas_type"] = "zero"
    scheduler_config["use_karras_sigmas"] = True
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
    pipe.enable_vae_tiling()
    _cached_inpaint_pipe = pipe
    return pipe


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def make_face_mask(size, feather=30):
    """Create a soft elliptical mask for context-aware face inpainting."""
    w, h = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    mx, my = int(w * 0.12), int(h * 0.10)
    draw.ellipse([mx, my, w - mx, h - my], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=feather))


def prepare_canny(pil_img, target_size, mode="loose"):
    """
    Convert template image into Canny edges for ControlNet conditioning.
    Three modes trade layout strictness for creative freedom:
        - loose:  heavy blur + high thresholds  (silhouette only)
        - normal: moderate blur + standard thresholds
        - tight:  light blur + low thresholds   (captures all detail)
    """
    img = pil_img.convert("RGB").resize(target_size, Image.LANCZOS)
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    if mode == "loose":
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(gray, 120, 220)
    elif mode == "tight":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 60, 160)
    else:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 100, 200)

    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def save_output(image, prompt, style, face_count, seed, model_id):
    """Save final image with JSON metadata for experimental logging."""
    save_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_{style.replace(' ', '_')}"
    image.save(os.path.join(save_dir, f"{name}.png"))
    with open(os.path.join(save_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts, "style": style, "prompt": prompt,
            "seed": seed, "detected_faces": face_count,
            "base_model": model_id,
        }, f, ensure_ascii=False, indent=4)
    print(f"Saved: {save_dir}/{name}.png")


def resolve_template(template_choice, custom_upload):
    """Return the active template PIL.Image based on user selection."""
    if template_choice == "Custom upload":
        return custom_upload
    filename = BUILTIN_TEMPLATES.get(template_choice)
    if filename is None:
        return None
    path = os.path.join(TEMPLATE_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: built-in template not found at {path}")
        return None
    return Image.open(path)


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------
def build_prompts(user_prompt, style):
    """
    Compose positive and negative prompts.
    Layout instruction is placed first to survive CLIP's 77-token cap
    even when user descriptions are verbose.
    """
    config = MODEL_CONFIGS[style]

    if style == "2D Anime":
        style_pos = "anime illustration, vibrant colors, key visual, cel-shaded"
        face_strength = 0.40
    else:
        # AAA game concept art style with strong cinematic atmosphere
        # Use () weighting to push lighting/atmosphere terms harder
        style_pos = (
            "AAA game concept art, epic fantasy character illustration, "
            "highly detailed realistic face and skin, "
            "intricate ornate armor with metalwork details, "
            "(dramatic cinematic lighting:1.3), (strong rim light:1.3), "
            "(moody atmosphere:1.2), warm-cool color contrast, "
            "soft gradient background, painterly background, "
            "rich fabric materials, leather and metal textures, "
            "digital painting masterpiece, artstation trending, "
            "hero shot promotional artwork"
        )
        face_strength = 0.40

    layout_part = (
        "character turnaround sheet, three full body T-pose views: "
        "front view, side profile view, back view, "
        "arms hanging down, neutral pose, "
        "pure white background"
    )

    prompt = (
        f"{layout_part}, "
        f"{user_prompt}, "
        f"{style_pos}, "
        f"{config['quality_pos']}"
    )

    negative_prompt = (
        f"{config['quality_neg']}, "
        # Block concrete scene elements but allow atmospheric color/lighting
        "ruined building, debris, rubble, industrial machinery, "
        "trees, forest, mountains, indoor room, furniture, "
        "ground texture, floor pattern, sky background, clouds, "
        "cluttered scene, props in background, multiple objects, "
        # No equipment display panels
        "equipment display, weapon rack, item showcase, gear layout, "
        "accessories displayed separately, inventory grid, item catalog, "
        # No props or weapons
        "weapons, sword, shield, banner, flag, heraldic decoration, "
        "ornamental props, decorative shields, "
        # No headless figures
        "headless figure, missing head, faceless body, empty hood, "
        # No grayscale aesthetic
        "grayscale, monochrome, desaturated, sepia, sketch only, lineart only, "
        # Identity and view consistency
        "multiple different characters, different outfits between views, "
        "two front views, missing back view, only one character, only two characters, "
        # No cropped figures
        "cropped body, missing legs, missing feet, "
        # Hand-deformity suppression (weighted)
        "(bad hands:1.3), (deformed fingers:1.3), (malformed hands:1.3), "
        "(fused fingers:1.2), (extra fingers:1.2), (missing fingers:1.2), "
        "(too many fingers:1.2), (mutated hands:1.2), poorly drawn hands, "
        # Pose constraints
        "dynamic pose, action pose, dancing pose, "
        "flying motion, jumping, sitting, walking, running, "
        "arms raised, arms crossed, hands behind back, "
        "asymmetric pose, dramatic angle, "
        # View angle constraints
        "three quarter view, partial side view, oblique view, "
        # Face quality
        "asymmetric face, asymmetric eyes, cross-eyed, lazy eye, "
        # Style purity
        f"{'3d render, photograph' if style == '2D Anime' else 'anime, manga, 2d cartoon, flat shading, cel-shaded, chibi, hand-drawn lineart, amateur snapshot, low quality photo'}"
    )
    return prompt, negative_prompt, face_strength


# ----------------------------------------------------------------------
# Main generation pipeline
# ----------------------------------------------------------------------
def generate_character(
    prompt, style, template_choice, custom_template,
    seed=60, cfg=6.5, steps=35,
    cn_scale=0.55, cn_end=0.55,
    canny_mode="loose",
    min_face_ratio=0.06,
    enable_inpaint=True,
    inpaint_strength_override=0.0,
):
    seed = int(seed)
    generator = torch.Generator("cuda").manual_seed(seed)
    print(f"\nGenerating | {style} | seed={seed} | CFG={cfg}")

    base_pipe = load_base_pipe(style)

    if not prompt or not prompt.strip():
        prompt = "a young adult character, simple outfit"

    full_prompt, neg_prompt, face_strength = build_prompts(prompt, style)
    word_count = len(full_prompt.split())
    if word_count > 77:
        print(f"  Prompt: {word_count} tokens (exceeds CLIP cap; style words may be truncated)")
    else:
        print(f"  Prompt: {word_count} tokens")

    # Resolve template (built-in or custom)
    template_img = resolve_template(template_choice, custom_template)

    if template_img is not None:
        control_image = prepare_canny(template_img, (OUT_W, OUT_H), mode=canny_mode)
        active_cn = float(cn_scale)
        print(f"  Template: {template_choice} | Canny: {canny_mode} | CN: {cn_scale}/{cn_end}")
    else:
        control_image = Image.new("RGB", (OUT_W, OUT_H), (0, 0, 0))
        active_cn = 0.0
        print("  No template - text-only generation (layout will be unstable)")

    # Stage 1: ControlNet-guided base generation
    print("[Stage 1] Base generation...")
    base_output = base_pipe(
        prompt=full_prompt,
        negative_prompt=neg_prompt,
        image=control_image,
        controlnet_conditioning_scale=active_cn,
        control_guidance_end=float(cn_end),
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        generator=generator,
        width=OUT_W, height=OUT_H,
    ).images[0]
    torch.cuda.empty_cache()

    # Stage 2: YOLO face detection
    print("[Stage 2] Face detection...")
    cv_img = cv2.cvtColor(np.array(base_output), cv2.COLOR_RGB2BGR)
    results = face_detector(cv_img, conf=0.35, verbose=False)
    raw_boxes = (results[0].boxes.xyxy.cpu().numpy()
                 if len(results) > 0 and results[0].boxes is not None
                 else [])

    # Filter out faces smaller than threshold (template artifacts, etc.)
    min_size = OUT_W * float(min_face_ratio)
    boxes = [b for b in raw_boxes
             if max(b[2] - b[0], b[3] - b[1]) >= min_size]
    print(f"  Detected {len(boxes)} valid faces (filtered from {len(raw_boxes)})")

    final_img = base_output.copy()

    # Stage 3: Context-aware face refinement
    if enable_inpaint and len(boxes) > 0:
        if inpaint_strength_override > 0:
            face_strength = float(inpaint_strength_override)
        print(f"[Stage 3] Face refinement (strength={face_strength})...")
        inpaint_pipe = load_inpaint_pipe()
        inpaint_gen = torch.Generator("cuda").manual_seed(seed)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            face_w = x2 - x1
            face_h = y2 - y1

            # Context-aware crop with 25-pixel surrounding padding
            context_pad = 25
            x1p = max(0, x1 - context_pad)
            y1p = max(0, y1 - context_pad)
            x2p = min(OUT_W, x2 + context_pad)
            y2p = min(OUT_H, y2 + context_pad)
            context_img = final_img.crop((x1p, y1p, x2p, y2p))
            roi_size = context_img.size

            # Build elliptical soft mask in cropped coordinates
            local_x1 = x1 - x1p
            local_y1 = y1 - y1p
            local_x2 = x2 - x1p
            local_y2 = y2 - y1p
            pad_small = int(max(face_w, face_h) * 0.15)
            mask_roi = Image.new("L", roi_size, 0)
            draw = ImageDraw.Draw(mask_roi)
            draw.ellipse([
                max(0, local_x1 - pad_small),
                max(0, local_y1 - pad_small),
                min(roi_size[0], local_x2 + pad_small),
                min(roi_size[1], local_y2 + pad_small),
            ], fill=255)
            mask_roi = mask_roi.filter(ImageFilter.GaussianBlur(radius=15))

            # Upsample to SDXL native resolution for inpainting
            context_1024 = context_img.resize((1024, 1024), Image.LANCZOS)
            mask_1024 = mask_roi.resize((1024, 1024), Image.LANCZOS)

            face_prompt = (
                f"{prompt}, detailed face, expressive eyes, same character, "
                f"{'anime style' if style == '2D Anime' else 'semi-realistic detailed face, fantasy concept art'}"
            )
            face_neg = "deformed face, asymmetric eyes, blurry, different person, grayscale"

            fixed_1024 = inpaint_pipe(
                prompt=face_prompt,
                negative_prompt=face_neg,
                image=context_1024,
                mask_image=mask_1024,
                num_inference_steps=25,
                strength=face_strength,
                guidance_scale=7.0,
                generator=inpaint_gen,
                width=1024, height=1024,
            ).images[0]

            # Resize back and blend with soft mask
            fixed_context = fixed_1024.resize(roi_size, Image.LANCZOS)
            final_img.paste(fixed_context, (x1p, y1p), mask_roi)

    save_output(final_img, prompt, style, len(boxes), seed,
                MODEL_CONFIGS[style]["base"])
    torch.cuda.empty_cache()
    print("Done.\n")
    return final_img


# ----------------------------------------------------------------------
# Gradio interface
# ----------------------------------------------------------------------
with gr.Blocks(title="Character Three-View Generator") as demo:
    gr.Markdown("# Character Three-View Generator")
    gr.Markdown(
        "Generate character turnaround sheets (front, side, back T-pose views) "
        "from a text description using a three-stage generative AI pipeline "
        "(SDXL ControlNet → YOLOv8 face detection → SDXL Inpainting). "
        "Choose between **Animagine XL 3.1** for anime style or **DreamShaper XL** "
        "for stylized 3D concept art."
    )

    with gr.Row():
        with gr.Column(scale=1):
            p_input = gr.Textbox(
                label="Character description",
                placeholder=(
                    "e.g. a female warrior, long silver hair, "
                    "purple mage robe with gold trim, blue crystal staff"
                ),
                lines=4,
            )
            s_input = gr.Radio(
                ["2D Anime", "3D Stylized"],
                label="Style",
                value="2D Anime",
            )

            template_choice = gr.Radio(
                list(BUILTIN_TEMPLATES.keys()),
                label="Template",
                value="Female (default)",
            )
            custom_template = gr.Image(
                label="Custom template (only used when 'Custom upload' is selected)",
                type="pil",
                visible=False,
            )

            # Show/hide custom upload based on selection
            def _toggle_upload(choice):
                return gr.update(visible=(choice == "Custom upload"))
            template_choice.change(
                fn=_toggle_upload,
                inputs=template_choice,
                outputs=custom_template,
            )

            with gr.Accordion("ControlNet settings", open=True):
                canny_mode_in = gr.Radio(
                    ["loose", "normal", "tight"],
                    value="loose",
                    label="Canny strictness",
                )
                cn_scale_in = gr.Slider(0, 1, value=0.55, step=0.05,
                                        label="ControlNet conditioning scale")
                cn_end_in   = gr.Slider(0.2, 1.0, value=0.55, step=0.05,
                                        label="ControlNet guidance end")
                cfg_input   = gr.Slider(3, 12, value=6.5, step=0.5,
                                        label="CFG scale")

            with gr.Accordion("Advanced", open=False):
                seed_input   = gr.Slider(0, 999999, value=60,  step=1,   label="Seed")
                steps_input  = gr.Slider(20, 60,    value=35,  step=1,   label="Inference steps")
                min_face_in  = gr.Slider(0.02, 0.20, value=0.06, step=0.01,
                                         label="Minimum face size ratio")
                inpaint_in   = gr.Checkbox(value=True, label="Enable face refinement")
                inpaint_strength_in = gr.Slider(
                    0.0, 0.8, value=0.0, step=0.05,
                    label="Face refinement strength (0 = automatic)",
                )

            btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Image(label="Output")

    btn.click(
        fn=generate_character,
        inputs=[p_input, s_input, template_choice, custom_template,
                seed_input, cfg_input, steps_input,
                cn_scale_in, cn_end_in,
                canny_mode_in, min_face_in, inpaint_in,
                inpaint_strength_in],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()
