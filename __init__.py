from .inpaint_cropandstitch import InpaintCropImproved
from .inpaint_cropandstitch import InpaintStitchImproved
from .inpaint_cropandstitch import StitcherDebugInfo
from .inpaint_cropandstitch import StitcherDebugImages
from .inpaint_cropandstitch import TemporalExpand
from .inpaint_cropandstitch import TemporalCollapse

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

    # OLD (deprecated)
    "InpaintCrop": "(OLD 💀) Inpaint Crop",
    "InpaintStitch": "(OLD 💀) Inpaint Stitch",
    "InpaintExtendOutpaint": "(OLD 💀) Extend Image for Outpainting",
    "InpaintResize": "(OLD 💀) Resize Image Before Inpainting",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
