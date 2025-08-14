# NanoDefectNet
End-to-end deep learning app for detecting defects in semiconductor wafers

Install pytorch, torchvision via `pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118` - Issues when trying to install via environment.yaml file

Add the project root ie, Folder containing this README to PYTHONPATH whichever way you want. One way would be to create a .env and write the following in it
```
PYTHONPATH=\full\path\to\projectroot
```
And place this .env file in the project root. Works for VS Code.

Another option would be to run `$env:PYTHONPATH = \full\path\to\projectroot` in powershell to set the env variable and then run the scripts.

Execute `pip install ipykernel==6.30.1` beforing running the jupyter notebooks. And in case of `ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed` error run `pip install jupyter`

##### Preprocessing

Run `python scripts/data_preprocess.py`

##### Data Augmentation

Run `python scripts/augment_train_data.py`

##### Training

For Resnet model:
Run `python .\nanodefectnet\run_train_test.py --path_config_file .\configs\classifier_resnet152_aug.yaml`
