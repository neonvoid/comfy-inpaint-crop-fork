from .inpaint_cropandstitch import InpaintCropImproved
from .inpaint_cropandstitch import InpaintStitchImproved
from .inpaint_cropandstitch import StitcherDebugInfo
from .inpaint_cropandstitch import StitcherDebugImages
from .inpaint_cropandstitch import TemporalExpand
from .inpaint_cropandstitch import TemporalCollapse
from .inpaint_cropandstitch import MaskRegionAnalyzer
from .inpaint_cropandstitch import MaskPlayerFilter
from .inpaint_cropandstitch import MaskColorizer
from .inpaint_cropandstitch import MaskRegionLimiter

# OLD
from .inpaint_cropandstitch_old import InpaintCrop
from .inpaint_cropandstitch_old import InpaintStitch
from .inpaint_cropandstitch_old import InpaintExtendOutpaint
from .inpaint_cropandstitch_old import InpaintResize

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "NV_InpaintCrop": InpaintCropImproved,
    "NV_InpaintStitch": InpaintStitchImproved,
    "NV_StitcherDebugInfo": StitcherDebugInfo,
    "NV_StitcherDebugImages": StitcherDebugImages,
    "NV_TemporalExpand": TemporalExpand,
    "NV_TemporalCollapse": TemporalCollapse,
    "NV_MaskRegionAnalyzer": MaskRegionAnalyzer,
    "NV_MaskPlayerFilter": MaskPlayerFilter,
    "NV_MaskColorizer": MaskColorizer,
    "NV_MaskRegionLimiter": MaskRegionLimiter,

    # OLD (deprecated)
    "InpaintCrop": InpaintCrop,
    "InpaintStitch": InpaintStitch,
    "InpaintExtendOutpaint": InpaintExtendOutpaint,
    "InpaintResize": InpaintResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_InpaintCrop": "NV ✂️ Inpaint Crop",
    "NV_InpaintStitch": "NV ✂️ Inpaint Stitch",
    "NV_StitcherDebugInfo": "NV ✂️ Stitcher Debug Info",
    "NV_StitcherDebugImages": "NV ✂️ Stitcher Debug Images",
    "NV_TemporalExpand": "NV ✂️ Temporal Expand",
    "NV_TemporalCollapse": "NV ✂️ Temporal Collapse",
    "NV_MaskRegionAnalyzer": "NV ✂️ Mask Region Analyzer",
    "NV_MaskPlayerFilter": "NV ✂️ Mask Player Filter",
    "NV_MaskColorizer": "NV ✂️ Mask Colorizer",
    "NV_MaskRegionLimiter": "NV ✂️ Mask Region Limiter",

    # OLD (deprecated)
    "InpaintCrop": "(OLD 💀) Inpaint Crop",
    "InpaintStitch": "(OLD 💀) Inpaint Stitch",
    "InpaintExtendOutpaint": "(OLD 💀) Extend Image for Outpainting",
    "InpaintResize": "(OLD 💀) Resize Image Before Inpainting",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
