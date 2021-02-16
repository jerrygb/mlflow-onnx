import click
import onnxruntime
import requests
import torch
import cloudpickle
import mlflow
import onnx
import os
import pandas as pd
import numpy as np
import mlflow.pyfunc
import yaml
from tqdm import tqdm
from transformers import RobertaTokenizer
from sys import version_info

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

# If python 3.8.6 is found, revert to 3.8.5 since 3.8.6 is not available.
if PYTHON_VERSION == '3.8.6':
    PYTHON_VERSION = '3.8.5'

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
                'mlflow',
                'onnx=={}'.format(onnx.__version__),
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'pandas=={}'.format(pd.__version__),
                'numpy=={}'.format(np.__version__),
            ],
        },
    ],
    'name': 'pytorch_env'
}

saved_path = 'roberta-saved'
classification_model = 'roberta-sequence-classification-9.onnx'
data_path = 'data'
artifacts = {
    "bert_model": os.path.join(os.getcwd(), saved_path, "model", "roberta-sequence-classification-9.onnx"),
    "bert_model_base_tokenizer": os.path.join(os.getcwd(), saved_path, "base", "roberta-base-11.onnx"),
    "metadata": os.path.join(os.getcwd(), data_path, "metadata.yaml")
}

TEMPLATE_BASE = "https://github.com/onnx/models/raw/{}/text/machine_comprehension/roberta/model/roberta-base-11.onnx"
TEMPLATE_MODEL = "https://github.com/onnx/models/raw/{}/text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx"


# Define the Models Class
class BertWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        with open(context.artifacts['metadata'], 'r') as f:
            metadata = yaml.safe_load(f)
        if metadata:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                                              revision=metadata['bert_model_base_tokenizer_revision'])
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # TODO: Unused code. Start using this as opposed to model binaries directly from Huggingface Model repository
        self.tokenizer_session = onnxruntime.InferenceSession(context.artifacts["bert_model_base_tokenizer"])
        self.ort_session = onnxruntime.InferenceSession(context.artifacts["bert_model"])

    @classmethod
    def to_numpy(cls, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    '''
    Main Entry point for the MLflow Pyfunc Prediction Class
    '''

    def predict(self, context, model_input):
        df = pd.DataFrame(columns=['text', 'sentiment'])
        # TODO: Convert to batch
        if isinstance(model_input, pd.core.frame.DataFrame):
            for index, row in model_input.iterrows():
                txt = row['text']
                self.predict_sentiment(df, txt)
        else:
            txt = model_input['txt']
            self.predict_sentiment(df, txt)

        # Return sentiments
        return df

    def predict_sentiment(self, df, txt):
        # Preprocess
        ort_inputs = self.preprocess(txt)

        # Prediction
        ort_out = self.ort_session.run(None, ort_inputs)

        # Postprocess
        self.postprocess(df, txt, ort_out)

    def postprocess(self, df, txt, ort_out):
        pred = np.argmax(ort_out)
        if (pred == 0):
            df = df.append({'text': txt, 'sentiment': 'negative'}, ignore_index=True)
        elif (pred == 1):
            df = df.append({'text': txt, 'sentiment': 'positive'}, ignore_index=True)


    def preprocess(self, txt):
        input_ids = torch.tensor(self.tokenizer.encode(txt, add_special_tokens=True)).unsqueeze(0)
        ort_inputs = {self.ort_session.get_inputs()[0].name: BertWrapper.to_numpy(input_ids)}
        return ort_inputs


def download(url, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


@click.command(
    help="Downloads the RoBERTa model and tokenizer and logs them into mlflow"
         "The model(s), tokenizer and its tags are logged with mlflow."
)
@click.option("--roberta-base-version", type=click.STRING, default='841d321fa51acc648d605a33c87ebe6726da8153',
    help="RoBERTa Tokenizer Version")
@click.option(
    "--roberta-model-version", type=click.STRING, default='bf6e73c4c68db02dc9cecd631a4a03a453932de0',
    help="RoBERTa Sentence Classification Model Version")
@click.option(
    "--upstream-roberta-base-version", type=click.STRING, default='841d321',
    help="Huggingface Model Repository revision. Used only when downloading directly using Huggingface/Transformers API"
)
def run(roberta_base_version, roberta_model_version, upstream_roberta_base_version):
    global my_run_id

    # TODO: Perhaps test loading and saving of the model
    # model = onnx.load(classification_model)
    # mlflow.onnx.save_model(model=model, path=saved_path, artifact_path='model')

    with mlflow.start_run(run_name='RoBERTa'):
        mlflow.set_tag('model_flavor', 'pytorch')
        mlflow.log_metric('accuracy', 0.99)

        mlflow.set_tag('roberta-base', roberta_base_version)
        mlflow.set_tag('roberta-sentence-classifier', roberta_model_version)

        if not os.path.exists(data_path):
            os.makedirs(data_path)
        metadata = {'bert_model_base_tokenizer_revision': upstream_roberta_base_version}
        with open(os.path.join(data_path, "metadata.yaml"), "w") as f:
            yaml.safe_dump(metadata, stream=f)

        # Download the base tokenizer and model
        base_url = TEMPLATE_BASE.format(roberta_base_version)
        model_url = TEMPLATE_MODEL.format(roberta_model_version)
        download(base_url, artifacts['bert_model_base_tokenizer'])
        download(model_url, artifacts['bert_model'])

        # Save the MLflow Model
        mlflow.pyfunc.log_model(artifact_path='model', python_model=BertWrapper(),
                                artifacts=artifacts, conda_env=conda_env)
        my_run_id = mlflow.active_run().info.run_id


# MLflow Tracking
# TODO: Add version tracking for base tokenizer and bert model
if __name__ == "__main__":
    run()

# Following is purely test code
# MLflow Models
model_uri = f'runs:/{my_run_id}/model'
sentence_classifier = mlflow.pyfunc.load_model(model_uri=model_uri)
