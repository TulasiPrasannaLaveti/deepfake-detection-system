"""Deepfake Detection System.

Pipeline: video -> frame extraction -> face detection -> face crop ->
hand-crafted features -> scikit-learn classifier -> video verdict.
"""
