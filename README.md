
# MRI-Super-Resolution

This repository contains code, data, and utilities for performing super-resolution on MRI images, specifically focused on improving the resolution from 3T to 7T MRI scans. The project leverages diffusion models and other machine learning techniques.


<img src="https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif" width="40" height="40" />

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Sample Data](#sample-data)
- [Utilities](#utilities)
- [License](#license)

## Project Structure

- **assets/**: Contains auxiliary files, such as videos or images used in the project.
  - `mri_slices_combined-ezgif.com-video-to-gif-converter.avi`: A video file likely showing combined MRI slices.

- **data/**: Contains the datasets used in the project.
  - **Brats2020/**: Dataset related to the BraTS 2020 competition, focusing on brain tumor segmentation.
  - **HCP/**: Dataset from the Human Connectome Project (HCP), used for MRI data.
  - **MNIST/**: Contains the MNIST dataset, possibly used for experimentation or model testing.

- **sample_data/**: Contains sample data for testing or demo purposes.
  - **Structural Preprocessed for 7T (1.6mm/59k mesh)**: Preprocessed structural data intended for use with 7T MRI resolution.
    - `100610_3T_Structural_1.6mm_preproc.zip`: A ZIP file containing the preprocessed data for 3T MRI.
    - `100610_3T_Structural_1.6mm_preproc.zip.md5`: MD5 checksum for the ZIP file to verify its integrity.

- **utils/**: Utility scripts and notebooks for the project.
  - `DDPM.ipynb`: A Jupyter notebook for training or experimenting with the DDPM model.
  - `diffusion_model_mnist.pth`: A pre-trained diffusion model on the MNIST dataset.
  - `eda_notebook.ipynb`: An exploratory data analysis (EDA) notebook.
  - `me.jpg`: An image file, possibly a personal photo or a sample image for testing.
  - `mri_slices_combined.avi`: Another video file, likely related to MRI data visualization.
  - `plot_t1w.py`: A Python script for plotting T1-weighted MRI images.

- **venv/**: Virtual environment for managing project dependencies.

## Installation

To set up the environment for this project:

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/your-username/MRI-Super-Resolution.git
   \`\`\`
2. Navigate to the project directory:
   \`\`\`bash
   cd MRI-Super-Resolution
   \`\`\`
3. Create a virtual environment:
   \`\`\`bash
   python3 -m venv venv
   \`\`\`
4. Activate the virtual environment:
   \`\`\`bash
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   \`\`\`
5. Install the required packages:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Usage

Instructions on how to run the notebooks, train models, and process the MRI data:

1. **Training the Model:**
   Open `DDPM.ipynb` in Jupyter Notebook and run the cells to start training the diffusion model.

2. **Exploratory Data Analysis:**
   Use `eda_notebook.ipynb` to explore the dataset before training.

3. **Visualization:**
   Run `plot_t1w.py` to generate visualizations of T1-weighted MRI images.

## Data

- **Brats2020**: Use for brain tumor segmentation tasks.
- **HCP**: Contains MRI data from the Human Connectome Project.
- **MNIST**: Can be used for model testing or as an additional dataset for experiments.

## Sample Data

Sample data under `sample_data/Structural Preprocessed for 7T` is provided to test the pipeline without downloading full datasets.

## Utilities

- **Scripts and notebooks**: Utility files provided in the `utils/` directory to assist with model training, data exploration, and visualization.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
