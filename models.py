from rvc.lib.tools.prerequisites_download import prerequisites_download_pipeline


print("downloading models...")
prerequisites_download_pipeline(models=True, exe=True)
