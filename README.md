# Contrastive_for_ADNI

To Run Autoencoder model
python run_ae.py --config {your config file}

To Run SimCLR model
- Pretraining
python run.py --config {your config file}
- Fintunning
python run_finetune.py --config {your config file}

Currently support only resnet models, we will support vision transfomer and DINO Learning
