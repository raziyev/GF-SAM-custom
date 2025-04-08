# GF-SAM (Customized Fork)

> **Fork Notice**: Modified version of [GF-SAM](https://github.com/ANDYZAQ/GF-SAM/tree/master).  
> **Key Changes**:  
> - Added query file support for targeted segmentation  
> - Optimized support image selection (no duplicates)

---

## Features

### 1. Query File Input
Process only specific images using `query_file.txt`:

#### Format:
```plaintext
folder_name  image_indices  # (without .jpg extension)
```

#### Example:

    bus 6 7 8  
    hotel_slipper 1 2 3  

- Predicts segmentation masks only for bus/6.jpg, bus/7.jpg, bus/8.jpg and hotel_slipper/1.jpg, etc.   


### 2. Unique Support Images

- Fixed [repeated image selection](https://github.com/ANDYZAQ/GF-SAM/blob/master/matcher/data/fss.py#L90) in the support set.

- Now guarantees unique images for better consistency.

## Usage

### 1. Add your query_file.txt to datasets/FSS-1000/

- Use the [format above](#format).

### 2. Run the model:

- Default (uses query_file.txt):

```bash
python main_eval.py --benchmark fss --nshot 1 --fold 0 --log-root "output/coco/fold0"
```

- Custom filename:

```bash
python main_eval.py --benchmark fss --queryfile my_query.txt --nshot 1 --fold 0 --log-root "output/coco/fold0"
```