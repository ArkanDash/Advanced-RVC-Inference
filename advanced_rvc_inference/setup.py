from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-rvc-inference",
    version="0.1.0",
    author="ArkanDash",
    description="A state-of-the-art web UI for rapid and effortless inference using RVC with ultimate vocal remover",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArkanDash/Advanced-RVC-Inference",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "advanced-rvc-inference=advanced_rvc_inference.app:main",
        ],
    },
)