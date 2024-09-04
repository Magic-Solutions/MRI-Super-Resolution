
# MRI-Super-Resolution

This repository focuses on enhancing the resolution from 1.5T to 3T MRI scans. By leveraging diffusion models, low-resolution MRI scans are upgraded to high-resolution images, enabling clearer, more detailed visualizations for medical imaging.

<table>
  <tr>
    <td>
      <img src="https://github.com/Magic-Solutions/MRI-Super-Resolution/blob/main/assets/mri_slice_LR.gif?v=1" width="600" height="200" />
    </td>
    <td style="vertical-align: middle; text-align: left; padding-left: 20px;">
      <span style="font-size: 16px;">
        Low resolution 1.5T MRI<br />
        3mm isotropic voxels
      </span>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Magic-Solutions/MRI-Super-Resolution/blob/main/assets/mri_slice_HR.gif?v=2" width="600" height="200" />
    </td>
    <td style="vertical-align: middle; text-align: left; padding-left: 20px;">
      <span style="font-size: 16px;">
        High resolution 3T MRI<br />
        1.6mm isotropic voxels
      </span>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Magic-Solutions/MRI-Super-Resolution/blob/main/assets/cosine_noise_scheduler.png?v=1" width="600" height="400" />
    </td>
    <td style="vertical-align: middle; text-align: left; padding-left: 20px;">
      <span style="font-size: 16px;">
        Cosine noise schedule
      </span>
    </td>
  </tr>
</table>



## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Sample Data](#sample-data)
- [Utilities](#utilities)
- [Credits](#credits)
- [License](#license)

## Project Structure

- **data/**: Contains the datasets used in the project.
  - **train/**: Contains the training data.
    - **low_res/**: Low-resolution MRI scans.
    - **high_res/**: High-resolution MRI scans.

- **models/**: Contains the model definitions.
  - `Unet_PyTorch.py`: 3D-Unet implementation in PyTorch.
  - `Unet_JAX.py`: 3D-Unet implementation in JAX.

- **training/**: Contains training scripts.
  - `train_jax.py`: Script for training the model using JAX.
  - `train_pytorch.py`: Script for training the model using PyTorch.

- **utils/**: Utility scripts and notebooks for data processing, model training, and visualization.
  - `DDPM.ipynb`: Jupyter notebook for training or experimenting with the DDPM model.
  - `plot_t1w.py`: Script for visualizing T1-weighted MRI images.

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

1. **Training the Model with JAX:**
   - Use `train_jax.py` to train the 3D-Unet model with JAX.
   ```bash
   python training/train_jax.py
2. **Visualization:**
   Run `plot_t1w.py` to generate visualizations of T1-weighted MRI images.

## Data

- **Brats2020**: Use for brain tumor segmentation tasks.
- **HCP**: Contains MRI data from the Human Connectome Project.
- **MNIST**: Can be used for model testing or as an additional dataset for experiments.

## Sample Data

Sample data under `sample_data/Structural Preprocessed for 7T` is provided to test the pipeline without downloading full datasets.

## Utilities

- **Scripts and notebooks**: Utility files provided in the `utils/` directory to assist with model training, data exploration, and visualization.

## Credits

This project is conducted in collaboration with GENCI, who provides the computational resources. Authored by Hendrik Chiche, Ludovic Corcos, and Logan Rouge.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
