# NanoDefectNet
End-to-end deep learning app for detecting defects in semiconductor wafers

Install pytorch, torchvision via `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118` - Issues when trying to install via environment.yaml file

Add the project root ie, Folder containing this README to PYTHONPATH whichever way you want. One way would be to create a .env and write the following in it
```
PYTHONPATH=\full\path\to\projectroot
```
And place this .env file in the project root. Works for VS Code.

Another option would be to run `$env:PYTHONPATH = \full\path\to\projectroot` in powershell to set the env variable and then run the scripts.

Execute `pip install ipykernel==6.30.1` beforing running the jupyter notebooks. And in case of `ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed` error run `pip install jupyter`

##### Sanity

Before committing changes run `pre-commit run --all-files` or `pre-commit run --file <file1>, <file2> ...`

##### Preprocessing

Run `python .\nanodefectnet\scripts\data_preprocess.py`

##### Data Augmentation

Run `python .\nanodefectnet\scripts\augment_train_data.py`

##### Training

For ResNet model:
Run `python .\nanodefectnet\run_train_test.py --path_config_file .\configs\classifier_resnet152_aug.yaml`

##### Inference

For ResNet model:
Run `python .\nanodefectnet\run_infer.py --model_name=ResNet152 --image=assets/test_images/center_defect.png --path_infer_config_file=configs/inference/infer.yaml`

#### API

Start the REST server using `uvicorn nanodefectnet.server.main:app --reload`

Test using windows powershell:  `curl.exe -X POST "http://127.0.0.1:8000/api/predict-waferdefect?model_name=ResNet152" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@D:/Computer Vision/Projects/NanoDefectNet/assets/test_images/center_defect.png"`

#### Docker (for API inference only and NOT for training/validation/test pipeline)

Build the image using `docker build -t nanodefectnet-app -f deploy/Dockerfile.serve .`

Run docker image using `docker run --gpus all -p 8000:8000 nanodefectnet-app`

Test using windows powershell:  `curl.exe -X POST "http://127.0.0.1:8000/api/predict-waferdefect?model_name=ResNet152" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@D:/Computer Vision/Projects/NanoDefectNet/assets/test_images/center_defect.png"`


#### Testing

Run only unit tests - `pytest -m unittest`

Run only integration tests - `pytest -m integration`

Run only tests that can be run on CI - `pytest -m runonci`

Run ALL tests - `pytest`

Note : Anytime a pytest marker is added to a pytest, ensure it is registered in `pytest.ini` otherwise pytest will complain

#### Data Augmentation using GenAI

1. Configure accelerate (only once). Choose desired parameters.

```bash
accelerate config
```

2. Run LoRA fine tuning

```bash
accelerate launch .\nanodefectnet\scripts\GenAI\fine_tune_sd_with_lora.py
```

3. Once LoRA weights have been saved, use it to generate augmented dataset

```bash
python .\nanodefectnet\scripts\GenAI\augment_train_dataset_genai.py
```
