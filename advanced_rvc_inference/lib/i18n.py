"""
Internationalization (i18n) Module for Advanced RVC Inference
Fallback implementation for internationalization support
Version 4.0.0

Authors: ArkanDash & BF667
Last Updated: November 26, 2025
"""

# I18n fallback implementation
class I18nAuto:
    def __init__(self):
        self.translations = {}
    
    def __call__(self, key):
        return key
    
    def __getitem__(self, key):
        return key
