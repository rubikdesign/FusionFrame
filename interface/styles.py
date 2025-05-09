#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSS styles for FusionFrame UI, compatible with Gradio 4.19.0
"""

CSS_STYLES = """
.container { max-width: 1200px; margin: auto; }
.image-preview { min-height: 400px; max-height: 600px; }
.error { color: red; }
.progress-area { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
.controls { display: flex; gap: 10px; margin-bottom: 15px; }
.info-panel { background: #e6f7ff; padding: 10px; border-radius: 4px; margin-top: 10px; }
/* Stiluri de buton simplificate */
#generate-btn {
    background: #4CAF50;
    border: none;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    font-weight: bold;
}
"""