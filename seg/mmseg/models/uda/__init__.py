# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add MinEnt and AdvSeg
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.advseg import AdvSeg
from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.dacs_meta_pseudo_label import DACS_META_PSLBL
from mmseg.models.uda.dacs_meta_eman import DACS_META_EMAN
from mmseg.models.uda.dacs_dynamic_masking import DACS_Dynamic_Masking
from mmseg.models.uda.minent import MinEnt

__all__ = [
    "DACS",
    "MinEnt",
    "AdvSeg",
    "DACS_META_PSLBL",
    "DACS_META_EMAN",
    "DACS_Dynamic_Masking",
]
