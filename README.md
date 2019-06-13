# MLFD
Machine Learning for Functional Dependencies

## Installation
Clone the project and cd into the folder.

```
git clone https://github.com/Larmor27/MLFD_fd_detection
cd MLFD_fd_detection
pip install datawig sklearn numpy pandas
```

If you want to perform FD-detection as well, load metanome binaries from separate repository.
```
git clone https://github.com/Larmor27/MLFD_fd_detection.git
```

Analysis has been moved to `fd_imputer_analysis.ipynb` within the root-folder of the project. Run `jupyter notebook` in the root directory to start a jupyter notebook server and access the notebook.

## Detect FDs
At this point, FD detection has to be performed manually using Metanome. To do so, follow the instructions in the MLFD_fd_detection repository.
