"""Minimal DEIM registry for v12 + ALPF inference."""
from .deim import DEIM, DualDecoderDEIM, DualModalDEIM
from .dfine_decoder import DFINETransformer
from .hybrid_encoder import HybridEncoder
from .postprocessor import PostProcessor
