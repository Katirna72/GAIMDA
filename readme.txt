GAIMDA

Description
PyTorch implementation of GAIMDA, a gated attention-based model for microbe-drug association prediction. It achieves state-of-the-art performance on MDAD and aBiofilm datasets via outer product convolutional decoding and dynamic negative sampling.

Dataset Information
Supported datasets:
- MDAD: 1373 drugs, 173 microbes, 2470 known associations
- aBiofilm: 1720 drugs, 140 microbes, 2884 known associations

Place dataset files under `./data/{dataset_name}/`, including the interaction matrix and optional structural/functional similarity files.

 Code Information
- main.py: 5-fold cross-validation, training loop and evaluation
- GAT.py: GAIMDA main model and CNN decoder
- layers.py: Gated graph attention layer
- datadeal.py: Similarity calculation and data preprocessing utilities
- utils.py: Evaluation metrics and graph normalization

Requirements
- Python 3.8+
- PyTorch, NumPy, SciPy, scikit-learn, pandas

Install dependencies:
pip install torch numpy scipy scikit-learn pandas

Usage Instructions
1. Put dataset files into the `data` directory
2. Adjust `DATASET_NAME` and hyperparameters in `main.py`
3. Run training:
python main.py

Default parameters: seed=42, hidden dim=128, embedding dim=64, 4 attention heads, dropout=0.5, lr=0.0001, KNN k=15, 30000 training steps, batch size=256.

Methodology
- Preprocessing: GIP kernel similarity, zero-padded safe feature construction, masked KNN heterogeneous graph, infinite dynamic negative sampling
- Encoder: Two-layer multi-head gated graph attention network
- Decoder: Outer product + 2D CNN for high-order interaction modeling
- Training: AdamW optimizer, BCEWithLogitsLoss, 5-fold cross-validation

Citations
If you use this code in your work, please cite the GAIMDA paper and the original MDAD / aBiofilm datasets.

License
MIT License 
