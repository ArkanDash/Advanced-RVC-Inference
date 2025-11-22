import os
import sys
import time
import logging
import warnings
import webbrowser

from tensorboard import program

sys.path.append(os.getcwd())

from main.app.variables import config, translations, logger

def launch_tensorboard():
    warnings.filterwarnings("ignore")
    for l in ["root", "tensorboard"]:
        logging.getLogger(l).setLevel(logging.ERROR)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", config.configs["logs_path"], f"--port={config.configs['tensorboard_port']}"])
    url = tb.launch()

    logger.info(f"{translations['tensorboard_url']}: {url}")
    if "--open" in sys.argv: webbrowser.open(url)

    return f"{translations['tensorboard_url']}: {url}"

if __name__ == "__main__": 
    launch_tensorboard()

    while 1:
        time.sleep(5)