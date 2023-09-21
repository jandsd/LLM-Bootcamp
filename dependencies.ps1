#run using "./req2.ps1" in project folder

# List of Python dependencies
$dependencies = @(
    "aiohttp==3.8.5",
    "aiosignal==1.3.1",
    "altair==4.2.2",
    "async-timeout==4.0.3",
    "attrs==23.1.0",
    "blinker==1.6.2",
    "cachetools==5.3.1",
    "certifi==2023.7.22",
    "charset-normalizer==3.2.0",
    "click==8.1.7",
    "colorama==0.4.6",
    "dataclasses-json==0.5.14",
    "easyocr==1.7.0",
    "entrypoints==0.4",
    "faiss-cpu==1.7.4",
    "filelock==3.12.2",
    "frozenlist==1.4.0",
    "gitdb==4.0.10",
    "GitPython==3.1.32",
    "greenlet==2.0.2",
    "idna==3.4",
    "imageio==2.31.2",
    "importlib-metadata==6.8.0",
    "Jinja2==3.1.2",
    "jsonschema==4.19.0",
    "jsonschema-specifications==2023.7.1",
    "langchain==0.0.177",
    "lazy_loader==0.3",
    "markdown-it-py==3.0.0",
    "MarkupSafe==2.1.3",
    "marshmallow==3.20.1",
    "mdurl==0.1.2",
    "mpmath==1.3.0",
    "multidict==6.0.4",
    "mypy-extensions==1.0.0",
    "networkx==3.1",
    "ninja==1.11.1",
    "numexpr==2.8.5",
    "numpy==1.25.2",
    "openai==0.27.10",
    "openapi-schema-pydantic==1.2.4",
    "opencv-python-headless==4.8.0.76",
    "packaging==23.1",
    "pandas==2.1.0",
    "Pillow==10.0.0",
    "protobuf==3.20.3",
    "pyarrow==13.0.0",
    "pyclipper==1.3.0.post4",
    "pydantic==1.10.12",
    "pydeck==0.8.1b0",
    "Pygments==2.16.1",
    "Pympler==1.0.1",
    "python-bidi==0.4.2",
    "python-dateutil==2.8.2",
    "python-dotenv==1.0.0",
    "pytz==2023.3",
    "PyWavelets==1.4.1",
    "PyYAML==6.0.1",
    "redis==5.0.0",
    "referencing==0.30.2",
    "regex==2023.8.8",
    "requests==2.31.0",
    "rich==13.5.2",
    "rpds-py==0.10.0",
    "scikit-image==0.21.0",
    "scipy==1.11.2",
    "shapely==2.0.1",
    "six==1.16.0",
    "smmap==5.0.0",
    "SQLAlchemy==2.0.20",
    "streamlit==1.22.0",
    "streamlit-chat==0.0.2.2",
    "sympy==1.12",
    "tenacity==8.2.3",
    "tifffile==2023.8.30",
    "tiktoken==0.4.0",
    "toml==0.10.2",
    "toolz==0.12.0",
    "torch==2.0.1",
    "torchvision==0.15.2",
    "tornado==6.3.3",
    "tqdm==4.66.1",
    "typing-inspect==0.8.0",
    "typing_extensions==4.5.0",
    "tzdata==2023.3",
    "tzlocal==5.0.1",
    "urllib3==2.0.4",
    "validators==0.21.2",
    "watchdog==3.0.0",
    "yarl==1.9.2",
    "youtube-transcript-api==0.6.1",
    "zipp==3.16.2"
)

# Loop through each dependency and install it
foreach ($dependency in $dependencies) {
    Write-Host "Installing $dependency"
    pip install $dependency
}

Write-Host "All Python dependencies installed successfully."
