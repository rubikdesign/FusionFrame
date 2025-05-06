#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Punct de intrare principal pentru FusionFrame 2.0
"""

import os
import sys
import argparse
import logging

from config.app_config import AppConfig
from interface.ui import FusionFrameUI

def parse_arguments():
    """
    Analizează argumentele de linie de comandă
    
    Returns:
        Argumentele analizate
    """
    parser = argparse.ArgumentParser(description="FusionFrame 2.0 - Advanced AI Image Editor")
    
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Create a public URL for the app"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port to run the app on"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to run the app on"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    parser.add_argument(
        "--low-vram", 
        action="store_true", 
        help="Enable low VRAM mode"
    )
    
    return parser.parse_args()

def main():
    """Funcția principală pentru rularea aplicației"""
    # Analizăm argumentele
    args = parse_arguments()
    
    # Configurăm logging-ul
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = AppConfig.setup_logging(level=log_level)
    
    # Configurări speciale
    if args.low_vram:
        logger.info("Low VRAM mode enabled")
        AppConfig.LOW_VRAM_MODE = True
    
    # Afișăm informații despre sistem
    logger.info(f"Starting FusionFrame {AppConfig.VERSION}")
    logger.info(f"Running on device: {AppConfig.DEVICE}")
    
    # Asigurăm că directoarele necesare există
    AppConfig.ensure_dirs()
    
    # Creăm și lansăm interfața
    try:
        ui = FusionFrameUI()
        logger.info(f"Launching interface on {args.host}:{args.port}")
        ui.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug
        )
    except Exception as e:
        logger.error(f"Error launching interface: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()