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

To split datasets and load an example, run `python fd_imputer.py`.

## Detect FDs
At this point, FD detection has to be performed manually using Metanome. To do so, follow the instructions in MLFD_fd_detection repository.
To load new FDs in the fd_imputer.py, update FD_PATH.
