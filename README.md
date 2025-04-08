# GF-SAM (Customized Fork)

> **Note**: This is an enhanced version of the original [GF-SAM](https://github.com/ANDYZAQ/GF-SAM/tree/master) with key improvements for Few-Shot Segmentation.

## Key Improvements
1. **Targeted Segmentation**: Process only specific images instead of the entire dataset.  
2. **Better Support Images**: No duplicate images in support sets.  

---
## Intallation

### Example conda environment setup
```python
conda create --name gfsam python=3.13.2
conda activate gfsam

# my CUDA version is 12.4
pip install torch==2.6.0 torchvision==0.21.0

pip install 'git+https://github.com/facebookresearch/detectron2.git'

git clone https://github.com/raziyev/GF-SAM-custom.git
cd GF-SAM-custom
pip install -r requirements.txt
```
## Model Preparation

- For model weights download, see original [GETTING_STARTED.md](https://github.com/ANDYZAQ/GF-SAM/blob/master/GETTING_STARTED.md).

## Dataset Preparation

#### 1. **Download**: 
Download datasets using the [official guide](https://github.com/ANDYZAQ/GF-SAM/tree/master/datasets).  

#### 2. **Folder Structure**:
Organize your files exactly like this:

```plain text
GF-SAM-custom/          
└── datasets/
    └── FSS-1000/
        ├── data/          # All images/masks go here
        │   ├── scratch/   # Example class folder
        │   │   ├── 1.jpg  # Image file (numeric names only)
        │   │   ├── 1.png  # Mask file (white=object, black=background)
        │   │   ├── 2.jpg 
        │   │   ├── 2.png
        │   │    ...       # more images
        │   ├── dent/      # Another class folder
        │   │   ├── 1.jpg 
        │   │   ├── 1.png
        │   │   ├── 2.jpg
        │   │   ├── 2.png
        │   │   ...
        │   ...
        ├── splits/        # Required for evaluation
        │    └── test.txt  # Controls which folders to process
        │
        └── query_file.txt # Your custom selection file (optional)
```

#### 3. **File Requirements**:

- **Images**:
    - Naming:
        - Only use numbers for filenames (e.g., 1.jpg, 2.png)
        - No letters/symbols in filenames

    - Types:
        - .jpg for input images
        - .png for mask files

    - Mask Colors:

        - Object: Pure white (RGB 255,255,255)
        - Background: Pure black (RGB 0,0,0)

- **test.txt configuration**:   
    - This file determines which class folders to process.
    - Only folders listed in test.txt will be segmented.

    - File location:  

    ```
    datasets/FSS-1000/splits/test.txt
    ```

    - File format:
    ```
    scratch
    dent
    # Add other folder names here, one per line
    ```    


- **query_file.txt configuration**: 

    - The `query_file.txt` lets you specify exactly which images to process, rather than segmenting all images in the dataset.   

    - File location:  

    ```
    datasets/FSS-1000/query_file.txt
    ```
    - File format:
    ```
    folder_name  image_number1 image_number2 ...  # (no .jpg extension)
    ```
    - Example:  
    Predicts segmentation masks only for dent/6.jpg, dent/7.jpg, dent/8.jpg and scratch/1.jpg, etc.    

    ```
    dent 6 7 8
    scratch 1 2 3
    ```

## Usage

### How It Works
1. **Input Selection**:
   - Processes only images specified in `query_file.txt`
   - Folders must be listed in `datasets/FSS-1000/splits/test.txt`

2. **Support Set Selection**:
   - Selects `--nshot` unique random images from remaining (query image excluded) images in the same folder
   - Ensures no duplicate support images (fixed from original implementation)

3. **Output**:
   - Implementation log file saved to `output/fss/fold0/`
   - Visualizations saved to `output/vis/` when `--visualize 1` is set

### Processing Modes
1. **Targeted Processing** (with `query_file.txt`):
   - Only processes specified images
   - Example: `bus 6 7 8` → processes `bus/6.jpg`, `bus/7.jpg`, `bus/8.jpg` as query

2. **Full Dataset Processing** (no query file):
   - Processes all images in folders listed in `test.txt`
   - Uses all images as queries (original behavior)

### Command Options

#### 1. Custom Query File
```bash
python main_eval.py \
    --datapath datasets \ 
    --benchmark fss \          
    --queryfile my_query.txt \  # Your custom selection (default= None)
    --nshot 1 \                # 1 support image per query
    --fold 0 \
    --log-root "output/fss/fold0" \
    --visualize 1              # Saves visualizations
```
#### 2. Process All Images
```bash
python main_eval.py \
    --datapath datasets \
    --benchmark fss \
    --nshot 1 \
    --fold 0 \
    --log-root "output/fss/fold0"
```

