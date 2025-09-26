# EFDiff: Frequency-Informed Diffusion for Extreme-Value Time Series Generation

This repository contains the implementation for the paper:  
**EFDiff: Frequency-Informed Diffusion for Extreme-Value Time Series Generation**  
(submitted to ICLR26).

---

## 1. Create your virtual environment
```bash
conda create -n EFDiff python=3.9
cd /path/to/EFDiff
```

## 2. Install Dependencies
```bash
conda activate EFDiff
pip install -r requirements.txt
```

## 3. Run the Main Pipeline
You can run the main training and evaluation pipeline with the provided configuration files.  
The dataset name must be chosen from the following list: **{stock, energy, etth, fmri, temperature}**.

```bash
# Training Process
python main.py --name <dataset> --config_file Config/<dataset>.yaml --gpu 0 --topk 5 --train

# Sampling Process
python main.py --name <dataset> --config_file Config/<dataset>.yaml --gpu 0 --sample 0 --topk 5 --milestone 10
```

### Example
```bash
# Training with stock dataset
python main.py --name stock --config_file Config/stock.yaml --gpu 0 --topk 5 --train

# Sampling with stock dataset
python main.py --name stock --config_file Config/stock.yaml --gpu 0 --sample 0 --topk 5 --milestone 10
```

## 4. Evaluation
All strategy algorithms are included in **Evaluation.ipynb**, and you can obtain the corresponding results simply by running it directly.
