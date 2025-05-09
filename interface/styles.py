#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Styles for Gradio in FusionFrame 2.0
"""

CSS_STYLES = """
.container { max-width: 1200px; margin: auto; }
.image-preview { min-height: 400px; max-height: 600px; }
.error { color: red; }
.progress-area { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
.controls { display: flex; gap: 10px; margin-bottom: 15px; }
.info-panel { background: #e6f7ff; padding: 10px; border-radius: 4px; margin-top: 10px; }
.example-btn { margin: 5px; }


#generate-btn {
    background: linear-gradient(135deg, #6e8efb, #a777e3);
    border: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    font-weight: bold;
}

#generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.12);
}

.example-btn {
    border-radius: 8px;
    border: 1px solid #ddd;
    background-color: #f9f9f9;
    transition: all 0.2s ease;
}

.example-btn:hover {
    background-color: #f0f0f0;
    border-color: #ccc;
    transform: scale(1.02);
}


@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

.progress-area:not(:empty) {
    animation: pulse 1.5s infinite;
}
"""