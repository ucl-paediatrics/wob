#!/bin/bash
# Setup script for use on Aridhia DRE or similar remote virtual environments
# Run this script using `bash -i path/to/setup.sh`
# Interactive bash mode (-i) is required for the conda commands to work properly. 
#
# We use conda because without it uv fails due to conflicing builds in the existing conda environment on the DRE.
#
# Once built, activate the kernel in jupyter notebooks using the kernel named "Python (beacon)" from the Kernel menu.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELATIVE_PATH="$(python -c "import os.path; print(os.path.relpath('$SCRIPT_DIR'))")"
PYTHON_VERSION="3.11"

echo "Setting up environment from  $RELATIVE_PATH..."
echo "Creating conda env 'wob' with python $PYTHON_VERSION"
conda create -n wob python=$PYTHON_VERSION -y
echo "Activating environment"
conda activate wob
pip install -e $RELATIVE_PATH
echo "Creating ipython kernel for jupyter..."
python -m ipykernel install --user --name=wob --display-name "Python (wob)"
echo "Setup complete."