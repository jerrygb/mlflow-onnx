# Install the OS dependencies 

Install the following dependency if you are running a mac,

```
brew install libomp
```

Install conda binaries since mlflow primarily uses conda for dependency management.

# Python dependencies

The dependencies for training and inference is already handled using conda files. The following dependencies are only if you would like to test or debug parts of your programs outside of the mlflow cli.

```
# Setup a virtual env and install the dependencies like so
pip install -r requirements.txt
```

# Training/Tracking the Pre-Trained Models

If you take a look at the `MLProject` file, it details the main entry points for training/tracking code. In addition to entry points, it provides the critical parameters for the model training/tracking.

In our case, since we are loading a pretrained model, all we require is the revision tags for the encoder and the model.

We log our models using the mlflow cli as follows,

```bash
$ mlflow run . --experiment-name RoBERTa
/Users/jerry/.virtualenvs/nora/lib/python3.8/site-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (1.26.3) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
2021/02/16 13:39:12 INFO mlflow.projects.utils: === Created directory /var/folders/80/qytvyjvd46j5g4qd6jcbh7840000gn/T/tmp0ysx64ot for downloading remote URIs passed to arguments of type 'path' ===
2021/02/16 13:39:12 INFO mlflow.projects.backend.local: === Running command 'source /Users/jerry/miniconda/bin/../etc/profile.d/conda.sh && conda activate mlflow-e71be12bb108c541f87d085e45337fc5b1219f5a 1>&2 && p
ython main.py --roberta-base-version 2c317e2e4d34d80adc89cdc958d55e5cdf6cb06c --roberta-model-version bf6e73c4c68db02dc9cecd631a4a03a453932de0 --upstream-roberta-base-version 841d321' in run with ID 'afb816ebacd4
467aa436e887eb02774c' ===
/Users/jerry/.virtualenvs/nora/lib/python3.8/site-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (1.26.3) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/Users/jerry/mlflow-poc/roberta-saved/base/roberta-base-11.onnx: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 476M/476M [00:09<00:00, 53.9MiB/s]
/Users/jerry/mlflow-poc/roberta-saved/model/roberta-sequence-classification-9.onnx: 100%|███████████████████████████████████████████████████████████████████████████████████████| 476M/476M [00:09<00:00, 50.4MiB/s]
2021/02/16 13:40:25 INFO mlflow.projects: === Run (ID 'afb816ebacd4467aa436e887eb02774c') succeeded ===
```

Once we log models, metadata, tags, parameters, we can view the details on the MLflow Console.

Once completed, you should see the output mlflow run id. In the above run, the id is `afb816ebacd4467aa436e887eb02774c`. We will use this in the next section.

# Inference

For simplicity, let us just execute a batch inference/prediction using the run id we obtained earlier.

```
mlflow models predict -m runs:/afb816ebacd4467aa436e887eb02774c/model --input-path sample.csv --content-type csv
```

The above command does a batch inference and produces the output to *stdout* in **json** format.

# TODO

A few have been marked in code.

