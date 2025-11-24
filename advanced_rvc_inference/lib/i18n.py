# I18n fallback implementation
class I18nAuto:
    def __init__(self):
        self.translations = {}
    
    def __call__(self, key):
        return key
    
    def __getitem__(self, key):
        return key
