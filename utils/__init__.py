"""Utility modules for FusionFrame application."""

import logging

# Set up module logger
logger = logging.getLogger(__name__)

# NU importa modulele aici pentru a evita importurile circulare
# Codul original era: from . import face_utils, io_utils