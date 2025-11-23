import os
import sys
import inspect

sys.path.append(os.getcwd())

from main.library.speaker_diarization.speechbrain import fetch, run_on_main
from main.library.speaker_diarization.features import DEFAULT_TRANSFER_HOOKS, DEFAULT_LOAD_HOOKS

def get_default_hook(obj, default_hooks):
    for cls in inspect.getmro(type(obj)):
        if cls in default_hooks: return default_hooks[cls]
        
    return None

class Pretrainer:
    def __init__(self, loadables=None, paths=None, custom_hooks=None, conditions=None):
        self.loadables = {}

        if loadables is not None: self.add_loadables(loadables)
        self.paths = {}

        if paths is not None: self.add_paths(paths)
        self.custom_hooks = {}

        if custom_hooks is not None: self.add_custom_hooks(custom_hooks)
        self.conditions = {}

        if conditions is not None: self.add_conditions(conditions)
        self.is_local = []

    def add_loadables(self, loadables):
        self.loadables.update(loadables)

    def add_paths(self, paths):
        self.paths.update(paths)

    def add_custom_hooks(self, custom_hooks):
        self.custom_hooks.update(custom_hooks)

    def add_conditions(self, conditions):
        self.conditions.update(conditions)

    @staticmethod
    def split_path(path):
        def split(src):
            if "/" in src: return src.rsplit("/", maxsplit=1)
            else: return "./", src

        return split(path)

    def collect_files(self, default_source=None):
        loadable_paths = {}
        for name in self.loadables:
            if not self.is_loadable(name): continue
            save_filename = name + ".ckpt"

            if name in self.paths: source, filename = self.split_path(self.paths[name])
            elif default_source is not None:
                filename = save_filename
                source = default_source
            else: raise ValueError

            fetch_kwargs = {"filename": filename, "source": source}
            path = None

            def run_fetch(**kwargs):
                nonlocal path

                path = fetch(**kwargs)

            run_on_main(run_fetch, kwargs=fetch_kwargs, post_func=run_fetch, post_kwargs=fetch_kwargs)

            loadable_paths[name] = path
            self.paths[name] = str(path)
            self.is_local.append(name)

        return loadable_paths

    def is_loadable(self, name):
        if name not in self.conditions: return True
        condition = self.conditions[name]

        if callable(condition): return condition()
        else: return bool(condition)

    def load_collected(self):
        paramfiles = {}
        for name in self.loadables:
            if not self.is_loadable(name): continue

            if name in self.is_local: paramfiles[name] = self.paths[name]
            else: raise ValueError

        self._call_load_hooks(paramfiles)

    def _call_load_hooks(self, paramfiles):
        for name, obj in self.loadables.items():
            if not self.is_loadable(name): continue
            loadpath = paramfiles[name]

            if name in self.custom_hooks:
                self.custom_hooks[name](obj, loadpath)
                continue

            default_hook = get_default_hook(obj, DEFAULT_TRANSFER_HOOKS)
            
            if default_hook is not None:
                default_hook(obj, loadpath)
                continue

            default_hook = get_default_hook(obj, DEFAULT_LOAD_HOOKS)

            if default_hook is not None:
                end_of_epoch = False
                default_hook(obj, loadpath, end_of_epoch)
                continue

            raise RuntimeError