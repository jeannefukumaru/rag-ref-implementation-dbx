# RAG reference architecture implementation 
## Use case: RAG Chatbot based on Databricks Documentation
## Pre-requisites
The following components are required to deploy and run this chatbot:
- A [vector search endpoint](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/create-query-vector-search#create-a-vector-search-endpoint)
- A LLM-endpoint which will be used by our chatbot to generate responses. 
  Use any [provisioned throughput foundation model APIs](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-models/#--provisioned-throughput-foundation-model-apis) 
  e.g. a LLama2-7b-chat endpoint (version 3) from the Databricks Marketplace.
- An embedding model endpoint (e.g. bge-large from the Databricks Marketplace)
- A LLM-endpoint which will be used as a judge for the evaluation of the chatbot. 
  Use any high-quality model from the [provisioned throughput foundation model APIs](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-models/#--provisioned-throughput-foundation-model-apis) 
  e.g. a LLama2-70b-chat endpoint (version 3) from the Databricks Marketplace.


## Project Components
The chatbot project consists of the following two main components:
### Data Preparation
The data preparation part is responsible for loading the raw documents, processing them and storing the final text chunks in a vector database index in Unity Catalog.
We use a [medallion architecture](https://www.databricks.com/glossary/medallion-architecture):
- __Bronze__: We ingest the raw documents from a Unity Catalog Volume folder and store them in a bronze table. 
  Each document is represented as a row in a Delta table using binary types. The implementation is located in the [01_raw_to_bronze.py](./jeanne_chatbot/data/notebooks/01_raw_to_bronze.py) notebook.
- __Silver__: Silver tables store the parsed documents. 
- __Gold__: Gold tables store the final text chunks. We use the parsed documents from the silver tables to create the final text chunks. The current implementation
in [03_silver_to_gold.py](./jeanne_chatbot/data/notebooks/03_silver_to_gold.py) uses a simple [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)
provided by Langchain. More advanced chunking strategies can easily be added to this notebook.
- __Vector Search Index__: The text chunks are then stored in a vector database index in Unity Catalog. The index is used by the chatbot to retrieve the most similar text chunks
  to a given query. The index is created using the [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) functionality in the [04_gold_to_vs.py](./jeanne_chatbot/data/notebooks/04_gold_to_vs.py) notebook.

All the data preparation components are defined in specific notebooks and deployed as jobs:
- Relevant resources are located in [data](./jeanne_chatbot/data) and [data/notebooks](./jeanne_chatbot/data/notebooks).
- Associated jobs are defined in [data-assets.yml](./jeanne_chatbot/assets/data-assets.yml).

### Chatbot Creation, Validation & Deployment
After the data preparation part has been executed successfully, the chatbot creation, validation and deployment part can be executed.
- __Chatbot Creation__: The chatbot creation part is responsible for creating the chatbot model using Langchain, logging it to the Mlflow experiment and registering it in the Mlflow Registry using the stage `Staging`.
  The implementation is located in the [01_chatbot.py](./jeanne_chatbot/chatbot/notebooks/01_chatbot.py) notebook. We are using [LangChain Expression Language](https://python.langchain.com/docs/expression_language/) to allow
for quick experimentation and development of a chatbot relying on retrieval-augmented-generation (RAG). 
- __Deployment__: The final step is the deployment of the chatbot model to a [model-serving endpoint](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/). We use a CPU-based model-serving endpoint which is able to scale to 0 if not used.

All the chatbot-related components are defined in specific notebooks and deployed as jobs:
- Relevant resources are located in [chatbot](./jeanne_chatbot/chatbot) and [chatbot/notebooks](./jeanne_chatbot/chatbot/notebooks).
- Associated jobs are defined in [chatbot-assets.yml](./jeanne_chatbot/assets/chatbot-assets.yml).

### Tuning
If you have deployed a first version of the new chatbot and defined a set of evaluation questions in the [evaluation_data.json](./jeanne_chatbot/chatbot/data/evaluation_data.json) file you can now start tuning your chatbot.
If you want to test different set of configurations, e.g. different prompts, LLM endpoints or data-preparation parameters such as `chunk_size` you can follow the
approach implemented in the [tuning_end_2_end.py](./jeanne_chatbot/tuning/notebooks/tuning_end_2_end.py) notebook. This notebook is designed to be run interactively in the Databricks Workspace
using Databricks Repos ([see below](#### Work within the Databricks Workspace environment)) and will allow you to quickly test different configurations and evaluate the performance of your chatbot using Mlflow.


## Configuration
This chatbot project uses [Databricks Asset Bundles](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/bundles/). Configuration is
defined through variables which are declared in the [databricks.yml](./databricks.yml) file. The actual configuration is
specified by defining their values in [configuration.yml](./configuration.yml). All variables have been assigned a default value
during setup of this project but can be changed at any time. The configuration will be used to define the corresponding input
parameters of the jobs deployed to the different environment target, i.e. dev, staging and production. If notebooks are run
for interactive development outside of the jobs the configuration will not take effect. Note though, that the notebook widgets
may also have default values defined based on the initial setup of this project.

## Development Process
The following sections describe the development process for the different environments, i.e. `dev`, `staging` and `prod`.

### `Dev` Environment - Interactive Development
As the name suggests, the `dev` environment is used for interactive development, tuning and testing of the chatbot application.
In order to develop interactively you will have two options, i.e. to work and develop in an IDE or to use the Databricks notebook environment.
Both options require you to first create a feature-branch from e.g. the `main` branch where you will implement all your changes. If you just created the chatbot project using the 
databricks asset bundle template described above you can push the corresponding code to the newly created feature-branch.

#### Work with an IDE
If you want to work with an IDE you can now open the project folder and start implementing your changes or deploy a 
dev-version right away. To deploy the `dev` target from the IDE simply open a terminal and use databricks asset bundle's 
deployment functionality:
```bash
databricks bundle deploy --target dev
```
This will deploy all required resources to you `dev` environment/target, i.e. 
- Mlflow experiment (defined in [ml-artifact-assets.yml](./jeanne_chatbot/assets/ml-artifact-assets.yml))
- Model in the Mlflow Registry (defined in [ml-artifact-assets.yml](./jeanne_chatbot/assets/ml-artifact-assets.yml))
- Workflow jobs for data-preparation and chatbot deployment (prefixed with the project-name). These jobs are defined in through 
  [Databricks Asset Bundles](https://learn.microsoft.com/en-us/azure/databricks/workflows/jobs/how-to/use-bundles-with-jobs) in
  [data-assets.yml](./jeanne_chatbot/assets/data-assets.yml) and [chatbot-assets.yml](./jeanne_chatbot/assets/chatbot-assets.yml) respectively.
All resources will be prefixed with `dev` to differentiate them from the `prod` or `staging` resources which is important 
especially if you use the same databricks workspace for all environments.

You can now test your implementation by running the deployed jobs in the databricks workspace. You can either trigger
them manually through the UI or use `databricks bundle run <job-name> -t dev` to start the jobs from the terminal. `<job-name>` refers
to either `dataprep_job` or `chatbot_job` as defined in the corresponding yml-files under [assets](./template/jeanne_chatbot/assets).

##### Example
Considering an example where we have created a project named `new_usecase_chatbot` using the [databricks_chatbot_factory](https://github.com/DigitalInnovation/databricks_chatbot_factory/tree/main) template. 
Let's assume we are happy with this first basic version of 
the `new_usecase_chatbot` project and want to deploy it to the `dev` environment. We simply execute the above-mentioned command
`databricks bundle deploy --target dev` and end up with the following resources which can be seen in the Databricks workspace:
- Mlflow experiment `dev_new_usecase_chatbot`
  ![Mlflow Experiment](./images/mlflow_experiment.png)
- Model registered in the Mlflow Registry `dev_new_usecase_chatbot`
  ![Mlflow Registry Model](./images/mlflow_model.png)
- Databricks jobs `dev_new_usecase_chatbot_dataprep` and `dev_new_usecase_chatbot_chain` under `Workflows`
  ![Databricks Jobs](./images/deployed_jobs.png)
  You can trigger these jobs
  manually, first the data-preparation one and then the chatbot one, to test your implementation. The data-preparation job will create bronze, silver
  and gold tables associated with the different stages of the data-preparation and finally the vector search index in the Unity Catalog and
  schema specified during setup. The chatbot job will create the chatbot model using Langchain, log it to the Mlflow experiment, register it in the Mlflow Registry
  and deploy and test a corresponding model serving endpoint.

#### Work within the Databricks Workspace environment
Working and implementing interactively in the Databricks Workspace allows you to directly test your changes to e.g. the Langchain-based
chatbot without deploying it to an endpoint. The workflow would look like the following:
- Create a [Databricks-Repo](https://learn.microsoft.com/en-us/azure/databricks/repos/) and checkout or create a feature-branch for your chatbot project.
- The main code is located in the `notebook` folders for both the [data-preparation](./template/jeanne_chatbot/data/notebooks) 
and the [chatbot](./template/jeanne_chatbot/data/notebooks) part.
- Input parameters are defined as [Databricks Notebook Widgets](https://learn.microsoft.com/en-us/azure/databricks/notebooks/widgets) 
and can be used to interactively test your implementation. While many of the default values will be pre-populated based on the information you provided
when creating the project you should make sure to provide the correct input parameters.
- Make sure to set valid names for Experiment, model, vector search index and table names (bronze, silver, gold) parameters. The default set for the interactive
  notebooks is always related to the project-name but won't add the suffixes `dev`, `staging` or `prod` which are used when deploying the jobs through databricks asset bundles.
- If they don't exist yet, make sure to create the Mlflow Experiment and the Mlflow Model specified as input parameters before executing the chatbot-based notebooks.
- Spin up an interactive cluster and run the notebooks to test your implementation.

#### Evaluation
The [01_chatbot.py](./template/jeanne_chatbot/chatbot/notebooks/01_chatbot.py.tmpl) notebook provided implements the 
chatbot chain using Langchain, logs and evaluates the model using the evaluation dataset. Hence, make sure to provide
relevant evaluation data in the [evaluation_data.json](./template/jeanne_chatbot/chatbot/data/evaluation_data.json) file. The default is just a dummy dataset.

Once you have defined proper evaluation data you should adjust two validation parameters used in the [validation notebook](jeanne_chatbot/chatbot/notebooks/02_validation.py)
- The `judge_metrics_threshold_production` parameter refers to the mean of all judge metrics, e.g. answer-similarity, faithfulness, etc.
- The `recall_threshold_metric_production` parameter defines the retriever-based threshold, currently it uses the `recall_at_3` metric.

The validation notebook will fail if any of the thresholds are not met. You can adjust the thresholds based on your requirements and the performance of your chatbot.
If the validation fails, the entire job will fail and the chatbot won't be deployed to the model-serving endpoint.

_Note, the current validation notebook is just a simple example of what is possible. Adjust it to your needs and requirements and add
additional validation steps to make sure your chatbot is performing as expected._

Irrespective of the type of development environment (IDE or Databricks Workspace), whenever you are happy with your 
implementation push your changes to the feature-branch in the remote repository and open a pull request to merge your changes into the `development` branch.

### `Staging` Environment - Deployment and testing
The `staging` environment is used for testing and validating the implementation end-2-end. It is associated with the `development` branch in the repository as defined
in the [`deployment_dev.yml`](template/.github/workflows/deployment_staging.yml.tmpl) file.

So once we are happy with our implementation in the feature branch created in the previous step we now want to test and validate it end-2-end.
To deploy the required components to the `staging` environment and perform some tests, do the following:

1. Create the `development` branch if it does not exist.
2. The repository defines some Github action [workflows](./template/.github/workflows) which will deploy the required resources to either the `staging` or
   `prod` environment based on the branch that is pushed to. Each workflow will use the databricks-cli to deploy the required resources and thus
   needs to be able to authenticate with the databricks workspace. This is done using a [github secret](https://docs.github.com/en/actions/security-guides/encrypted-secrets) called `DATABRICKS_TOKEN`.
   To create this secret, follow the instructions [here](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository).
   The `DATABRICKS_TOKEN` is a [personal access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html#generate-a-personal-access-token) for the databricks workspace.
_TODO: Using the personal access token is fine for testing and dev, however, consider using a service principal and associated m2m oauth authentication especially for prod._
3. Now, merge the feature-branch you created in the previous step to the `development` branch. This will trigger the deployment of the required resources to the `staging` environment. 
   Once the Github action for the deployment has successfully finished you should see the following resource in the databricks workspace. Each of the resource-names is prefixed with either `dev` or `prod` to differentiate between the environments.
   - Mlflow experiment with the name you specified during creation of the chatbot. 
   - Model registered in the Mlflow Registry with the name you specified during creation of the chatbot.
   - A databricks job `<target>-<project-name>_dataprep_staging` under `Workflows` which will load the required raw documents, 
     process them and store the final text chunks in a vector database index in Unity Catalog with the name `<catalog_name>.<schema_name>.<project-name>_vs_index_direct_staging`
     using the vector search endpoint defined during creation of the chatbot-project. Intermediate tables, i.e. Bronze, Silver and Gold tables are also created with 
     similar names `<catalog_name>.<schema_name>.<project-name>_bronze/silver/gold`.
   - A databricks job `<target>-<project-name>_chain` which will create the actual chatbot model using Langchain, log it to the Mlflow experiment and register it in the Mlflow Registry.
     In addition, it will use the [evaluation dataset](./template/jeanne_chatbot/chatbot/data/evaluation_data.json) to evaluate the chatbot. It uses functionality provided by `mlflow.evaluate()`
     to evaluate both the retriever part as well as the end-2-end chatbot using a LLM as a judge (the judge endpoint provided during creation).
4. To deploy your new chatbot to a `staging` model-serving endpoint that can be used to test & serve any 3rd-party tool or font-end application simply
   run the both job, i.e. first the `dataprep` and then the `chain` job. Once both finished successfully you should see a new model-serving endpoint being created with the name `<project_name>_<target>`.
   Once the endpoint is fully up and running (i.e in `ready` state) you can query it from any 3rd-party tool or font-end application. If you simply want to test it you can create a query from withing the
   databricks model-serving UI. This will also show you the required format of the payload.

_Note: Going forward is recommended to extend the `Staging` workflow in Github to not just deploy the resources but also to run the jobs, some tests and validation steps.
Once the tests are successful the `Staging` workflow on Github should delete the endpoint and all deployed resources again._


## `Prod` Environment - Production Deployment
While the `Staging` environment is used for testing and validating the implementation end-2-end, the `prod` environment is used for the actual _production_ deployment.
Here, we defined a Github action [workflow](.github/workflows/deployment_prod.yml) which will deploy the required resources to the `prod` environment once changes are merged into the `main` branch.
The Github-workflow will not only deploy the jobs but also run them and hence deploy a production model-serving endpoint that can be queried by any 3rd-party tool or font-end application.
All resources are suffixed with `prod` to differentiate them from the `staging` or `dev` resources which is especially important if you use the same databricks workspace for all environments.


## Where to go from here
- Make sure to define proper evaluation sets for your chatbot to be able to validate its performance.
- Using the evaluation data you can easily tune your chatbot to meet your requirements. I.e.
  - Test different prompts
  - Experiment with different components as part of you chatbot chain, e.g. use different retrieval strategies (mmr, score-threshold, etc.)
  - Test and evaluate different LLMs, e.g. served through different endpoints, or embedding models


## Additional Notes
- If the LLM endpoint is in a different workspace you'll need to specify the endpoint URL as you would with a [central model registry](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/multiple-workspaces#set-up-the-api-token-for-a-remote-registry).
  Only personal access tokens are supported in this case.
- Databricks Vector search supports [two modes of authentication](https://docs.databricks.com/en/generative-ai/vector-search.html#data-protection-and-authentication).
  We currently use personal access tokens. For a proper production deployment consider using service principals.