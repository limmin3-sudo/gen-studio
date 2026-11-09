import json
import struct

from genstudio import workflows
from genstudio.registry import Registry, is_full_checkpoint

REG = Registry.load(__import__("pathlib").Path(__file__).parent / "fixtures" / "models.yaml")
ZIT = REG.get("cyberrealistic-zit")
ILL = REG.get("realism-illustrious")


def classes(g):
    return [n["class_type"] for n in g.values()]


def assert_links_valid(g):
    for nid, node in g.items():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, f"node {nid} links to missing {v[0]}"
    assert set(classes(g)) <= set(workflows.REQUIRED_NODES)


def node(g, cls):
    return next(n["inputs"] for n in g.values() if n["class_type"] == cls)


def test_zimage_txt2img_matches_reference_settings():
    g, extra = workflows.txt2img(ZIT, "a portrait", "", 1024, 1024, 7)
    assert_links_valid(g)
    assert node(g, "UNETLoader")["unet_name"] == ZIT.file.name
    assert node(g, "CLIPLoader") == {"clip_name": "qwen_3_4b_fp8_mixed.safetensors", "type": "lumina2", "device": "default"}
    assert node(g, "ModelSamplingAuraFlow")["shift"] == 3
    ks = node(g, "KSampler")
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"]) == (8, 1.0, "res_multistep", "simple")
    assert "ConditioningZeroOut" in classes(g)
    assert "EmptySD3LatentImage" in classes(g)


def test_illustrious_txt2img_uses_checkpoint_clip_skip_and_negative():
    g, extra = workflows.txt2img(ILL, "a woman", "hat", 832, 1216, 1)
    assert_links_valid(g)
    assert node(g, "CheckpointLoaderSimple")["ckpt_name"] == ILL.file.name
    assert node(g, "CLIPSetLastLayer")["stop_at_clip_layer"] == -2
    texts = [n["inputs"]["text"] for n in g.values() if n["class_type"] == "CLIPTextEncode"]
    assert texts[0].startswith("99bsy99") and texts[0].endswith("a woman")
    assert texts[1].startswith("hat, lowres")
    assert node(g, "EmptyLatentImage")["width"] == 832


def test_inpaint_pastes_back_with_mask():
    g, _ = workflows.inpaint(ILL, "gen-studio/a.png", "gen-studio/m.png", "red ribbon", "", 3)
    assert_links_valid(g)
    for c in ("LoadImageMask", "GrowMask", "SetLatentNoiseMask", "ImageCompositeMasked", "DifferentialDiffusion"):
        assert c in classes(g)
    assert node(g, "LoadImageMask")["channel"] == "red"
    assert node(g, "KSampler")["denoise"] == 1.0


def test_img2img_denoise():
    g, _ = workflows.img2img(ZIT, "gen-studio/a.png", "", "", 3, 0.4)
    assert_links_valid(g)
    assert node(g, "KSampler")["denoise"] == 0.4


def _write_safetensors(path, keys):
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0")


def test_full_checkpoint_detection(tmp_path):
    full, unet = tmp_path / "full.safetensors", tmp_path / "unet.safetensors"
    _write_safetensors(full, ["model.diffusion_model.x", "first_stage_model.y", "text_encoders.qwen.z"])
    _write_safetensors(unet, ["model.diffusion_model.x", "layers.0.attn"])
    assert is_full_checkpoint(full) is True
    assert is_full_checkpoint(unet) is False


QWEN = REG.get("qwen-image-21-turbo")


def test_qwen21_turbo_matches_comfy_template():
    g, _ = workflows.txt2img(QWEN, 'poster that says "HELLO"', "", 1024, 1536, 5)
    assert_links_valid(g)
    assert node(g, "CLIPLoader")["type"] == "qwen_image"
    enc = node(g, "TextEncodeQwenImage21")
    assert enc["prompt"] == 'poster that says "HELLO"' and enc["resolution"] == 1024 and "vae" in enc
    assert node(g, "ManualSigmas")["sigmas"].startswith("1.0, 0.978453")
    assert node(g, "CFGGuider")["cfg"] == 1.0
    assert node(g, "KSamplerSelect")["sampler_name"] == "euler"
    assert node(g, "EmptyLatentImage")["height"] == 1536
    assert "KSampler" not in classes(g) and "QwenImage21Cache" in classes(g)


def test_qwen21_partial_denoise_trims_schedule():
    g, _ = workflows.img2img(QWEN, "gen-studio/a.png", "", "", 1, 0.5)
    assert_links_valid(g)
    assert node(g, "ManualSigmas")["sigmas"] == "0.5, 0.414568, 0"
    g, _ = workflows.inpaint(QWEN, "gen-studio/a.png", "gen-studio/m.png", "x", "", 1)
    assert_links_valid(g)
    assert node(g, "ManualSigmas")["sigmas"].startswith("1.0")


WAN = REG.get("wan22-14b")


def test_wan22_two_experts_like_comfy_template():
    g, _ = workflows.txt2img(WAN, "a woman by the window", "", 832, 1216, 9)
    assert_links_valid(g)
    unets = sorted(n["inputs"]["unet_name"] for n in g.values() if n["class_type"] == "UNETLoader")
    assert unets == ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"]
    assert node(g, "CLIPLoader")["type"] == "wan"
    assert node(g, "EmptyHunyuanLatentVideo")["length"] == 1
    ks = [n["inputs"] for n in g.values() if n["class_type"] == "KSamplerAdvanced"]
    assert [(k["start_at_step"], k["end_at_step"], k["add_noise"]) for k in ks] == [(0, 2, "enable"), (2, 10000, "disable")]
    assert all(k["steps"] == 4 and k["cfg"] == 1.0 for k in ks)
    loras = [n["inputs"]["lora_name"] for n in g.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert len(loras) == 2 and all("lightx2v" in x for x in loras)
    assert all(n["inputs"]["shift"] == 5 for n in g.values() if n["class_type"] == "ModelSamplingSD3")


def test_wan22_light_edit_uses_low_noise_only():
    g, _ = workflows.img2img(WAN, "gen-studio/a.png", "", "", 1, 0.4)
    assert_links_valid(g)
    ks = [n["inputs"] for n in g.values() if n["class_type"] == "KSamplerAdvanced"]
    assert len(ks) == 1 and ks[0]["start_at_step"] == 2 and ks[0]["add_noise"] == "enable"
    g, _ = workflows.inpaint(WAN, "gen-studio/a.png", "gen-studio/m.png", "x", "", 1)
    assert_links_valid(g)
    assert sum(1 for n in g.values() if n["class_type"] == "DifferentialDiffusion") == 2
