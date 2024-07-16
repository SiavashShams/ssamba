# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : hubconf.py

# Authors
# - Leo

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


# Frame-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssast_frame_base(refresh: bool = False, window_secs: float = 1.0, **kwargs):
    ckpt = _urls_to_filepaths(
        "https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1",
        refresh=refresh,
    )
    return _UpstreamExpert(ckpt, "base_f", window_secs)


# Patch-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssast_patch250_base(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/ssast_base_250.pth"
    return _UpstreamExpert(ckpt, "base_p", window_secs)

def ssast_patch300_base(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/ssast_base_300.pth"
    return _UpstreamExpert(ckpt, "base_p", window_secs)

def ssast_patch400_base(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/ssast_base_400.pth"
    return _UpstreamExpert(ckpt, "base_p", window_secs)


def ssast_patch250_small(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/ssast_small_250.pth"
    return _UpstreamExpert(ckpt, "small_p", window_secs)

def ssast_patch300_small(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/ssast_small_300.pth"
    return _UpstreamExpert(ckpt, "small_p", window_secs)

def ssast_patch400_small(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/ssast_small_400.pth"
    return _UpstreamExpert(ckpt, "small_p", window_secs)


def ssast_patch250_tiny(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/ssast_tiny_250.pth"
    return _UpstreamExpert(ckpt, "tiny_p", window_secs)

def ssast_patch300_tiny(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/ssast_tiny_300.pth"
    return _UpstreamExpert(ckpt, "tiny_p", window_secs)

def ssast_patch400_tiny(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/ssast_tiny_400.pth"
    return _UpstreamExpert(ckpt, "tiny_p", window_secs)



def amba_patch250_base(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/amba_base_250.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def amba_patch300_base(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/amba_base_300.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def amba_patch400_base(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/amba_base_400.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)


def amba_patch250_small(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/amba_small_250.pth"
    return _UpstreamExpert(ckpt, "small_a", window_secs)

def amba_patch300_small(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/amba_small_300.pth"
    return _UpstreamExpert(ckpt, "small_a", window_secs)

def amba_patch400_small(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/amba_small_400.pth"
    return _UpstreamExpert(ckpt, "small_a", window_secs)


def amba_patch250_tiny(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/amba_tiny_250.pth"
    return _UpstreamExpert(ckpt, "tiny_a", window_secs)

def amba_patch300_tiny(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/amba_tiny_300.pth"
    return _UpstreamExpert(ckpt, "tiny_a", window_secs)

def amba_patch400_tiny(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/engram/naplab/shared/ssast/models/amba_tiny_400.pth"
    return _UpstreamExpert(ckpt, "tiny_a", window_secs)



