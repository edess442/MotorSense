from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from torchvision.models import ResNet50_Weights, resnet50
try:
    import hiera  # pip install hiera-transformer
except Exception:
    hiera = None


def _strip_prefix(k: str, prefixes: Tuple[str, ...]) -> str:
    for p in prefixes:
        if k.startswith(p):
            return k[len(p):]
    return k


def _tensor_shape(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(x) for x in shape)


def _capture_load_report(module: nn.Module, incoming_state_dict: dict, load_result, source: str) -> dict[str, object]:
    target_state = module.state_dict()
    incoming_keys = list(incoming_state_dict.keys())
    target_keys = list(target_state.keys())

    loaded_keys = []
    mismatched_shape = []
    unexpected_keys = []
    for key, value in incoming_state_dict.items():
        incoming_shape = _tensor_shape(value)
        if key not in target_state:
            unexpected_keys.append(key)
            continue
        target_shape = _tensor_shape(target_state[key])
        if incoming_shape is None or target_shape is None:
            unexpected_keys.append(key)
            continue
        if incoming_shape == target_shape:
            loaded_keys.append(key)
        else:
            mismatched_shape.append(
                {
                    "key": key,
                    "checkpoint_shape": incoming_shape,
                    "model_shape": target_shape,
                }
            )

    missing_keys = list(getattr(load_result, "missing_keys", []))
    load_result_unexpected = list(getattr(load_result, "unexpected_keys", []))
    unexpected_key_set = set(unexpected_keys)
    unexpected_key_set.update(load_result_unexpected)

    return {
        "source": source,
        "module_class": module.__class__.__name__,
        "target_key_count": len(target_keys),
        "incoming_key_count": len(incoming_keys),
        "loaded_keys": sorted(loaded_keys),
        "missing_model_keys": sorted(missing_keys),
        "unexpected_checkpoint_keys": sorted(unexpected_key_set),
        "mismatched_shape_keys": sorted(mismatched_shape, key=lambda item: item["key"]),
    }


def _print_load_report(prefix: str, report: dict[str, object]) -> None:
    loaded_keys = list(report.get("loaded_keys", []))
    missing_keys = list(report.get("missing_model_keys", []))
    unexpected_keys = list(report.get("unexpected_checkpoint_keys", []))
    mismatched_shape = list(report.get("mismatched_shape_keys", []))
    print(
        f"{prefix} source={report.get('source', 'unknown')} "
        f"loaded={len(loaded_keys)} missing={len(missing_keys)} "
        f"unexpected={len(unexpected_keys)} mismatched={len(mismatched_shape)}"
    )


def _load_backbone_from_ckpt(backbone: nn.Module, ckpt_path: str) -> dict[str, object]:
    if ckpt_path is None:
        return {
            "source": None,
            "module_class": backbone.__class__.__name__,
            "target_key_count": len(backbone.state_dict()),
            "incoming_key_count": 0,
            "loaded_keys": [],
            "missing_model_keys": [],
            "unexpected_checkpoint_keys": [],
            "mismatched_shape_keys": [],
        }

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)

    cleaned = {}
    for k, v in sd.items():
        k2 = _strip_prefix(k, ("module.",))
        k2 = _strip_prefix(k2, ("backbone.", "model.", "net.", "encoder.", "trunk."))
        cleaned[k2] = v

    backbone_state = backbone.state_dict()
    loadable = {}
    for k, v in cleaned.items():
        if k in backbone_state and hasattr(v, "shape") and v.shape == backbone_state[k].shape:
            loadable[k] = v

    load_result = backbone.load_state_dict(loadable, strict=False)
    report = _capture_load_report(backbone, loadable, load_result, source=str(ckpt_path))
    _print_load_report("[backbone init]", report)
    return report


def _build_unknown_pretrained_report(source: str, module: nn.Module) -> dict[str, object]:
    return {
        "source": source,
        "module_class": module.__class__.__name__,
        "target_key_count": len(module.state_dict()),
        "incoming_key_count": 0,
        "loaded_keys": [],
        "missing_model_keys": [],
        "unexpected_checkpoint_keys": [],
        "mismatched_shape_keys": [],
        "note": "Could not capture internal checkpoint load details from the Hiera constructor.",
    }


def _apply_drop_path_linear_schedule(model: nn.Module, drop_path_rate: float) -> None:
    drop_path_rate = float(drop_path_rate)
    if drop_path_rate <= 0:
        return
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return
    depth = len(blocks)
    if depth == 0:
        return
    dpr = torch.linspace(0.0, drop_path_rate, depth).tolist()
    for i, blk in enumerate(blocks):
        if hasattr(blk, "drop_path"):
            dp = getattr(blk, "drop_path")
            if hasattr(dp, "drop_prob"):
                dp.drop_prob = float(dpr[i])
            elif hasattr(dp, "p"):
                dp.p = float(dpr[i])


def _instantiate_hiera_with_load_report(fn, *, variant: str, checkpoint: str, pretrained: bool, strict: bool, drop_path: float):
    if not pretrained:
        base = fn(
            num_classes=1,
            pretrained=False,
            checkpoint=str(checkpoint),
            strict=bool(strict),
            drop_path_rate=float(drop_path),
        )
        return base, {
            "source": None,
            "module_class": None,
            "target_key_count": len(base.state_dict()),
            "incoming_key_count": 0,
            "loaded_keys": [],
            "missing_model_keys": [],
            "unexpected_checkpoint_keys": [],
            "mismatched_shape_keys": [],
            "note": "No pretrained checkpoint requested.",
        }

    original_load_state_dict = nn.Module.load_state_dict
    captured_reports = []

    def patched_load_state_dict(self, state_dict, *args, **kwargs):
        result = original_load_state_dict(self, state_dict, *args, **kwargs)
        if isinstance(state_dict, dict):
            captured_reports.append(
                _capture_load_report(
                    self,
                    state_dict,
                    result,
                    source=f"hiera:{variant}:{checkpoint}",
                )
            )
        return result

    nn.Module.load_state_dict = patched_load_state_dict
    try:
        try:
            base = fn(
                num_classes=1,
                pretrained=True,
                checkpoint=str(checkpoint),
                strict=bool(strict),
                drop_path_rate=float(drop_path),
            )
        except TypeError:
            base = fn(
                num_classes=1,
                pretrained=True,
                checkpoint=str(checkpoint),
                strict=bool(strict),
            )
            _apply_drop_path_linear_schedule(base, float(drop_path))
    finally:
        nn.Module.load_state_dict = original_load_state_dict

    if not captured_reports:
        report = _build_unknown_pretrained_report(f"hiera:{variant}:{checkpoint}", base)
    else:
        report = max(captured_reports, key=lambda item: int(item.get("target_key_count", 0)))
    _print_load_report("[hiera pretrained init]", report)
    return base, report


class HieraVideoEPICClassifier(nn.Module):
    def __init__(
        self,
        variant: str = "hiera_base_16x224",
        checkpoint: str = "mae_k400_ft_k400",
        pretrained: bool = True,
        num_verbs: int = 97,
        num_nouns: int = 300,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        input_layout: str = "BTCHW",
        eval_returns_logits: bool = True,
        strict: bool = False,
        backbone_ckpt: Optional[str] = None,
        freeze_backbone: bool = False,
        finetune_last_layer_only: bool = False,
        last_layer_blocks: int = 1,
    ):
        super().__init__()
        if hiera is None:
            raise ImportError("hiera-transformer is required. Install with: pip install hiera-transformer")

        self.input_layout = str(input_layout).upper()
        self.num_verbs = int(num_verbs)
        self.num_nouns = int(num_nouns)

        if not hasattr(hiera, variant):
            raise ValueError(f"Unknown Hiera variant '{variant}'")
        fn = getattr(hiera, variant)
        base, self.pretrained_load_report = _instantiate_hiera_with_load_report(
            fn,
            variant=str(variant),
            checkpoint=str(checkpoint),
            pretrained=bool(pretrained),
            strict=bool(strict),
            drop_path=float(drop_path),
        )

        if eval_returns_logits and hasattr(base, "head") and hasattr(base.head, "act_func"):
            base.head.act_func = lambda x: x

        feat_dim = None
        if hasattr(base, "head") and hasattr(base.head, "projection") and isinstance(base.head.projection, nn.Linear):
            feat_dim = int(base.head.projection.in_features)

        self.backbone = base
        if backbone_ckpt:
            ckpt_path = Path(str(backbone_ckpt))
            if ckpt_path.exists():
                self.pretrained_load_report = _load_backbone_from_ckpt(self.backbone, str(ckpt_path))
            else:
                print(f"[emg_pretraining] backbone_ckpt not found, falling back to Hiera pretrained init: {ckpt_path}")

        self.has_forward_features = hasattr(self.backbone, "forward_features")

        if not self.has_forward_features:
            if feat_dim is None:
                raise RuntimeError(
                    "Could not infer feature dim. Your hiera package lacks forward_features and head.projection."
                )
            self.backbone.head.projection = nn.Identity()

        if feat_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 16, 224, 224)
                feats = self.backbone.forward_features(dummy)
                if feats.ndim != 2:
                    feats = feats.view(feats.size(0), -1)
                feat_dim = int(feats.size(1))

        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.verb_head = nn.Linear(feat_dim, self.num_verbs, bias=True)
        self.noun_head = nn.Linear(feat_dim, self.num_nouns, bias=True)

        for p in self.parameters():
            p.requires_grad = True

        def _enable_heads():
            for p in self.verb_head.parameters():
                p.requires_grad = True
            for p in self.noun_head.parameters():
                p.requires_grad = True

        if freeze_backbone or finetune_last_layer_only:
            for p in self.backbone.parameters():
                p.requires_grad = False
            _enable_heads()

        if finetune_last_layer_only:
            n = max(1, int(last_layer_blocks))
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is not None and len(blocks) > 0:
                for blk in list(blocks)[-n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
            else:
                last_mod = None
                for m in self.backbone.modules():
                    if any(True for _ in m.parameters(recurse=False)):
                        last_mod = m
                if last_mod is not None:
                    for p in last_mod.parameters():
                        p.requires_grad = True

            if not self.has_forward_features and hasattr(self.backbone, "head"):
                for p in self.backbone.head.parameters():
                    p.requires_grad = True

    def _to_bcthw(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_layout == "BCTHW":
            return x
        if self.input_layout == "BTCHW":
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"input_layout must be BTCHW or BCTHW, got {self.input_layout}")

    def _extract_backbone_tokens(self, vid: torch.Tensor) -> torch.Tensor:
        if self.has_forward_features:
            feats = self.backbone.forward_features(vid)
        else:
            _, intermediates = self.backbone(vid, return_intermediates=True)
            if not intermediates:
                raise RuntimeError("Backbone returned no intermediates.")
            feats = intermediates[-1]

        if feats.ndim == 2:
            return feats.unsqueeze(1)
        if feats.ndim < 3:
            raise RuntimeError(f"Unexpected backbone feature shape: {tuple(feats.shape)}")
        return feats.reshape(feats.size(0), -1, feats.size(-1))

    def _extract_backbone_features(self, vid: torch.Tensor) -> torch.Tensor:
        tokens = self._extract_backbone_tokens(vid)
        return tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        vid = self._to_bcthw(x)
        feats = self._extract_backbone_features(vid)

        feats = self.dropout(feats)
        return {
            "verb": self.verb_head(feats),
            "noun": self.noun_head(feats),
        }


class HieraVideoMultiHeadCodePredictor(nn.Module):
    def __init__(
        self,
        head_dims: dict[str, dict[str, int]],
        variant: str,
        checkpoint: str,
        pretrained: bool,
        strict: bool,
        dropout: float,
        drop_path: float,
        freeze_backbone: bool,
        input_layout: str,
        backbone_ckpt: str | None,
        finetune_last_layer_only: bool,
        last_layer_blocks: int,
        mode: str = "video",
    ):
        super().__init__()
        self.head_dims = {
            name: {
                "num_queries": int(spec["num_queries"]),
                "query_output_dim": int(spec["query_output_dim"]),
                "output_dim": int(spec["output_dim"]),
            }
            for name, spec in head_dims.items()
        }
        self.mode = str(mode).lower()
        if self.mode not in {"video", "image_pair"}:
            raise ValueError(f"Unsupported model mode '{mode}'. Expected one of: video, image_pair.")

        self.input_layout = str(input_layout).upper()
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self._to_bcthw = None
        self._extract_backbone_tokens = None

        if self.mode == "image_pair":
            self.backbone, backbone_report, feat_dim = self._build_image_pair_backbone(
                pretrained=pretrained,
                backbone_ckpt=backbone_ckpt,
                freeze_backbone=freeze_backbone,
                finetune_last_layer_only=finetune_last_layer_only,
                last_layer_blocks=last_layer_blocks,
            )
            self.has_forward_features = False
            token_dim = feat_dim * 2
        else:
            base = HieraVideoEPICClassifier(
                variant=variant,
                checkpoint=checkpoint,
                pretrained=pretrained,
                num_verbs=1,
                num_nouns=1,
                dropout=dropout,
                drop_path=drop_path,
                input_layout=input_layout,
                strict=strict,
                backbone_ckpt=backbone_ckpt,
                freeze_backbone=freeze_backbone,
                finetune_last_layer_only=finetune_last_layer_only,
                last_layer_blocks=last_layer_blocks,
            )
            self.backbone = base.backbone
            self.has_forward_features = bool(base.has_forward_features)
            self.dropout = base.dropout
            self._to_bcthw = base._to_bcthw
            self._extract_backbone_tokens = base._extract_backbone_tokens
            feat_dim = int(base.verb_head.in_features)
            token_dim = feat_dim
            backbone_report = getattr(base, "pretrained_load_report", None) or {}

        self.poolers = nn.ModuleDict(
            {
                name: AttentivePooler(
                    num_queries=int(spec["num_queries"]),
                    embed_dim=token_dim,
                    num_heads=min(8, max(1, token_dim // 64)),
                    mlp_ratio=4.0,
                    depth=1,
                )
                for name, spec in self.head_dims.items()
            }
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(token_dim, int(spec["query_output_dim"]))
                for name, spec in self.head_dims.items()
            }
        )
        head_keys = []
        for head_name, head in self.heads.items():
            for key in head.state_dict().keys():
                head_keys.append(f"heads.{head_name}.{key}")
        self.pretrained_load_report = {
            "mode": self.mode,
            "backbone": backbone_report,
            "new_head_keys": sorted(head_keys),
        }

    def _build_image_pair_backbone(
        self,
        pretrained: bool,
        backbone_ckpt: str | None,
        freeze_backbone: bool,
        finetune_last_layer_only: bool,
        last_layer_blocks: int,
    ) -> tuple[nn.Module, dict[str, object], int]:
        weights = ResNet50_Weights.IMAGENET1K_V2 if bool(pretrained) else None
        base = resnet50(weights=weights)
        backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        backbone_report = {
            "source": "torchvision:resnet50_imagenet1k_v2" if weights is not None else None,
            "module_class": backbone.__class__.__name__,
            "target_key_count": len(backbone.state_dict()),
            "incoming_key_count": len(backbone.state_dict()) if weights is not None else 0,
            "loaded_keys": sorted(backbone.state_dict().keys()) if weights is not None else [],
            "missing_model_keys": [],
            "unexpected_checkpoint_keys": [],
            "mismatched_shape_keys": [],
        }

        if backbone_ckpt:
            ckpt_path = Path(str(backbone_ckpt))
            if ckpt_path.exists():
                backbone_report = _load_backbone_from_ckpt(backbone, str(ckpt_path))
            else:
                print(f"[emg_pretraining] backbone_ckpt not found, falling back to torchvision ResNet init: {ckpt_path}")

        for p in backbone.parameters():
            p.requires_grad = True

        if freeze_backbone or finetune_last_layer_only:
            for p in backbone.parameters():
                p.requires_grad = False

        if finetune_last_layer_only:
            resnet_stages = [base.layer1, base.layer2, base.layer3, base.layer4]
            n = max(1, int(last_layer_blocks))
            for stage in resnet_stages[-n:]:
                for p in stage.parameters():
                    p.requires_grad = True

        return backbone, backbone_report, 2048

    def _extract_resnet_tokens(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image)
        if feats.ndim != 4:
            raise RuntimeError(f"Unexpected ResNet feature shape: {tuple(feats.shape)}")
        return feats.flatten(2).transpose(1, 2).contiguous()

    def _extract_image_pair_tokens(self, video: torch.Tensor) -> torch.Tensor:
        if video.size(1) < 2:
            raise ValueError(
                f"image_pair mode requires at least 2 frames so it can use first/last frames, got {video.size(1)}"
            )

        first_frame = video[:, 0, :, :, :]
        last_frame = video[:, -1, :, :, :]
        first_tokens = self._extract_resnet_tokens(first_frame)
        last_tokens = self._extract_resnet_tokens(last_frame)
        if first_tokens.shape[:2] != last_tokens.shape[:2]:
            raise RuntimeError(
                "First/last frame token shapes do not align for channelwise concatenation: "
                f"{tuple(first_tokens.shape)} vs {tuple(last_tokens.shape)}"
            )
        return torch.cat([first_tokens, last_tokens], dim=-1)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(video.shape)}")

        if self.mode == "image_pair":
            tokens = self._extract_image_pair_tokens(video)
        else:
            vid = self._to_bcthw(video)
            tokens = self._extract_backbone_tokens(vid)
        outputs = {}
        for name, head in self.heads.items():
            pooled = self.poolers[name](tokens)
            pooled = self.dropout(pooled)
            pred = head(pooled)
            outputs[name] = pred.reshape(pred.size(0), -1)
        return outputs


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)
            q = (xattn @ v)

        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x, mask=None):
        del mask
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        grid_size=None,
        grid_depth=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, return_attention=False, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.xattn = CrossAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class AttentivePooler(nn.Module):
    def __init__(
        self,
        num_queries: int = 1,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        norm_layer: nn.Module = nn.LayerNorm,
        init_std: float = 0.02,
        qkv_bias: bool = True,
        complete_block: bool = True,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
            )
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
            )

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for _ in range(depth - 1)
                ]
            )

        self.init_std = float(init_std)
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q
