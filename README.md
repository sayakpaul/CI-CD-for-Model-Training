# CI/CD for Model Training in ML

This repository holds files and scripts for incorporating simple CI/CD practices for model training in ML. Following
resources have been used to prepare the files under the `build` directory:

* [Vertex AI Training with TFX and Vertex Pipelines](https://www.tensorflow.org/tfx/tutorials/tfx/gcp/vertex_pipelines_vertex_training)
* [MLOps with Vertex AI by GCP](https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai)

Another accompanying repository for doing CI/CD for model training can be found here: [deep-diver/Model-Training-as-a-CI-CD-System](https://github.com/deep-diver/Model-Training-as-a-CI-CD-System). 

This repository acts as a sister repository to [deep-diver/Model-Training-as-a-CI-CD-System](https://github.com/deep-diver/Model-Training-as-a-CI-CD-System). It's recommended that one goes through this repository with the accompanying blog post from Google Cloud: [Model training as a CI/CD system: Part II](https://cloud.google.com/blog/topics/developers-practitioners/model-training-cicd-system-part-ii). The figures below schematically present what this project implements:

We first get a compiled pipeline specification for model re-training capable of accepting hyperparameter values as [`RuntimeParameters`](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/dsl/experimental/RuntimeParameter):

<p align="center">
  <img src=https://storage.googleapis.com/gweb-cloudblog-publish/images/image2_zJt72L0.max-900x900.png>
</p>

Then we schedule and trigger model training jobs:

<p align="center">
  <img src=https://storage.googleapis.com/gweb-cloudblog-publish/images/image6_79NQRAT.max-1700x1700.png width=550>
</p>

## Acknowledgements

[ML-GDE program](https://developers.google.com/programs/experts/) for providing GCP credits. Thanks to [Karl Weinmeister](https://twitter.com/kweinmeister?lang=hr) for providing review feedback on this project.
