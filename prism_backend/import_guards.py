"""
import_guards.py — Safe optional imports for third-party machine learning modules.
"""
try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    MinMaxScaler = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import cv2
except ImportError:
    cv2 = None
