# Name-Entity-Recognition


## Workflows

 - constants
 - config_entity
 - artifact_entity
 - components
 - pipeline
 - app.py



## Live matarials docs

[link](https://docs.google.com/document/d/1UFiHnyKRqgx8Lodsvdzu58LbVjdWHNf-uab2WmhE0A4/edit?usp=sharing)


## Git commands

```bash
git add .

git commit -m "Updated"

git push origin main
```


## AWS GCP Configuration

```bash
#Gcloud cli download link: https://cloud.google.com/sdk/docs/install#windows

gcloud init
```


## How to run?

```bash
conda create -n nerproj python=3.8 -y
```

```bash
conda activate nerproj
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```


## GCP:

- artifact registry
- line 41,49,73,53 in circleci config
- Opne circleci 


### Environment variables

GCLOUD_SERVICE_KEY --> service account

GOOGLE_COMPUTE_ZONE = asia-south1

GOOGLE_PROJECT_ID


- VM instances


50e54784c203bf6c8f523d341b717c28136442b42d5c9ab8dc2ddfd2975ba02e301f268242e67425

