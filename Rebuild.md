

# requirements
## c++
- x64 native tools shell for Visual Studio
- **********************************************************************
** Visual Studio 2022 Developer Command Prompt v17.14.7
** Copyright (c) 2025 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'

## conda

- call "C:\Program Files\Geosoft\Desktop Applications\python\condabin\conda.bat" activate
conda --version

## nvcc
- e.g. conda install -c nvidia cuda-toolkit=12.4

## package
- torch, ninja, rasterio

:: Ensure linker can find the Conda CUDA libs in this shell
set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"

:: Optional: ensure runtime DLLs resolve at import time
set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"


:: Clean cached build artifacts (your Torch doesn’t have clear_cache; manual delete is fine)
rmdir /s /q "%LOCALAPPDATA%\torch_extensions" 2>nul

:: Rebuild
python run_example.py

## timing
- compilation reports about 80s on this machne as per comment in code
