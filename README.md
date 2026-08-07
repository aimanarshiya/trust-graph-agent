# trust-graph-agent
TRUST-GRAPH
AI-POWERED FRAUD DETECTION, RISK SCORING, GRAPH ANALYSIS AND INTELLIGENT AGENTS

1. PROJECT OVERVIEW

Trust-Graph is an AI-powered fraud detection and trust analysis system designed to identify potentially fraudulent transactions and explain why a transaction is considered risky.

The system combines machine learning, transaction analysis, graph-based relationship analysis, risk scoring, large language model capabilities, and specialized AI agents. Instead of only producing a binary result such as "Fraud" or "Not Fraud", Trust-Graph is designed to answer four important questions:

1. Is the transaction risky?
2. Why is the transaction risky?
3. What entities or historical relationships are connected to the transaction?
4. What action should be taken based on the available evidence?

The project uses the IEEE-CIS Fraud Detection dataset from Kaggle as the primary machine-learning dataset. The machine-learning pipeline loads transaction and identity information, merges the datasets, performs categorical encoding and feature preparation, trains an XGBoost model, evaluates the model, and stores the resulting model artifacts.

After model training, the backend can use the trained model together with graph relationships, risk-scoring logic, and AI agents to produce an explainable fraud-risk assessment.

2. MAIN OBJECTIVE

The main objective of Trust-Graph is to build a fraud-analysis system that goes beyond simple transaction classification.

Traditional fraud detection may answer:

"Is this transaction fraudulent?"

Trust-Graph aims to additionally answer:

"Why is it suspicious?"
"What connected entities increase the risk?"
"What evidence supports the decision?"
"What should be done next?"

The system therefore combines predictive intelligence with relationship intelligence and explainability.

3. HIGH-LEVEL SYSTEM FLOW

The complete Trust-Graph workflow is:

Kaggle Dataset
        ↓
Dataset Download
        ↓
Transaction Data + Identity Data
        ↓
Data Loading
        ↓
Dataset Merging
        ↓
Categorical Feature Encoding
        ↓
Feature Preparation
        ↓
XGBoost Model Training
        ↓
Model Evaluation
        ↓
Model Artifacts
        ↓
Transaction Prediction
        ↓
Risk Scoring
        ↓
Graph Relationship Analysis
        ↓
Evidence Collection
        ↓
Explainer Agent
        ↓
Remediation Agent
        ↓
Self-Check Agent
        ↓
Final Risk Assessment

4. PROJECT DIRECTORY STRUCTURE

The current Trust-Graph project is organized into a main project directory containing the backend, machine-learning components, data, AI agents, database, graph engine, LLM integration, environment configuration, and Python virtual environment.

The main structure is:

TRUST-GRAPH/
    trust-graph-agent/
        backend/
            agents/
            data/
            external/
            ml/
                artifacts/
                train_model.py
            .env
            database.py
            graph_engine.py
            llm.py
            main.py
            risk_scorer.py
            trust_graph.db
        venv/
        .gitignore
        README.md

The exact contents may grow as additional features are implemented.

5. BACKEND FOLDER

The backend directory contains the main application logic of Trust-Graph.

It contains the AI agents, dataset files used by the application, machine-learning code and generated model artifacts, database logic, graph processing logic, LLM integration, application entry point, and risk-scoring functionality.

The backend is the central part of the system where the individual components communicate with each other.

6. AGENTS FOLDER

Location:

backend/agents/

The agents folder contains specialized intelligent agents used by Trust-Graph.

The current agents include:

explainer_agent.py
remediation_agent.py
selfcheck_agent.py

Each agent has a specific responsibility rather than making every component perform every task.

7. EXPLAINER AGENT

File:

backend/agents/explainer_agent.py

The Explainer Agent is responsible for converting technical fraud-detection signals into an understandable explanation.

The machine-learning model may produce a probability or prediction, but that prediction alone is not always useful to a human investigator.

The Explainer Agent is intended to explain the available evidence in human-readable language.

For example, an explanation may contain information such as:

The transaction has a high predicted fraud probability.
The transaction has unusual characteristics.
The device may be connected to multiple accounts.
The IP address may have suspicious historical relationships.
The transaction may be associated with a high-risk seller.

The exact evidence depends on the data and logic available to the system.

8. REMEDIATION AGENT

File:

backend/agents/remediation_agent.py

The Remediation Agent is responsible for suggesting actions after a transaction has been classified as risky.

The purpose is to convert a risk assessment into an actionable recommendation.

Possible recommendations may include:

Holding a transaction for manual review.
Requesting additional verification.
Reviewing the customer's previous activity.
Investigating connected devices or IP addresses.
Checking seller history.
Performing additional identity verification.

The recommendations should be based on the evidence and risk level generated by the system.

9. SELFCHECK AGENT

File:

backend/agents/selfcheck_agent.py

The Self-Check Agent is intended to validate the generated analysis before the final result is presented.

It can be used to check whether:

The explanation agrees with the risk assessment.
Important evidence has been considered.
The recommendations are consistent with the detected risk.
The final result is internally consistent.

The Self-Check Agent provides an additional validation layer between analysis and the final output.

10. DATA FOLDER

Location:

backend/data/

The data folder contains datasets used by the Trust-Graph application.

The current project structure contains files such as:

transactions.csv
sellers.csv
deliveries.csv

An identity dataset is also used by the machine-learning training pipeline.

11. TRANSACTIONS DATA

File:

backend/data/transactions.csv

The transactions dataset contains transaction-level information.

Transaction information is important for fraud detection because it contains attributes that describe individual financial or purchase activities.

The machine-learning model uses transaction information to learn patterns associated with fraudulent and legitimate transactions.

12. IDENTITY DATA

The identity dataset contains identity-related information associated with transactions.

Identity information can provide additional context that is not available from transaction information alone.

Identity-related attributes can help the system identify unusual patterns involving users, devices, addresses, email information, and other identity-related signals.

The machine-learning training script loads transaction data and identity data and combines them before training.

13. SELLERS DATA

File:

backend/data/sellers.csv

The sellers dataset contains seller-related information.

Seller information can be used to provide additional context about transactions and may help identify suspicious seller behavior or repeated transaction patterns.

14. DELIVERIES DATA

File:

backend/data/deliveries.csv

The deliveries dataset contains delivery-related information.

Delivery information can provide additional evidence related to shipping or delivery behavior.

It can potentially help identify repeated delivery destinations, unusual delivery patterns, or relationships between transactions and locations.

15. EXTERNAL FOLDER

Location:

backend/external/

The external folder is intended for external resources and externally obtained data.

This is a suitable location for datasets downloaded from external services such as Kaggle when the project implementation requires them to be stored separately from the application source code.

Large raw datasets should generally not be committed to GitHub.

16. MACHINE LEARNING FOLDER

Location:

backend/ml/

The ml folder contains the machine-learning implementation.

The main training script is:

backend/ml/train_model.py

The generated machine-learning artifacts are stored in:

backend/ml/artifacts/

17. TRAIN_MODEL.PY

File:

backend/ml/train_model.py

This is the main machine-learning training script.

The training script is responsible for loading the transaction and identity datasets, merging the datasets, preparing the features, encoding categorical columns, training the XGBoost model, evaluating the model, and saving the trained model and related information.

The training flow is approximately:

Load transaction data.
Load identity data.
Merge the datasets.
Identify categorical columns.
Encode categorical columns.
Prepare the training features.
Train XGBoost.
Generate predictions.
Calculate evaluation metrics.
Save the trained model.
Save feature-column information.
Save model metrics.

The current execution output shows:

Loading transaction data...
Loading identity data...
Merged shape: (590540, 434)
Encoding 31 categorical columns...
Final memory usage: 1.02 GB
Training XGBoost...

This means that the current merged training dataset contains approximately 590,540 rows and 434 columns before or around the final training preparation stage.

The pipeline also reports approximately 1.02 GB of memory usage during the shown execution stage.

18. MACHINE LEARNING ARTIFACTS

Location:

backend/ml/artifacts/

The artifacts folder contains files generated by the training process.

Typical generated files include:

fraud_model.joblib
feature_columns.joblib
model_metrics.joblib

19. FRAUD_MODEL.JOBLIB

File:

backend/ml/artifacts/fraud_model.joblib

This file stores the trained fraud-detection machine-learning model.

The trained model can later be loaded by the backend for prediction instead of training the model again every time the application starts.

20. FEATURE_COLUMNS.JOBLIB

File:

backend/ml/artifacts/feature_columns.joblib

This file stores the feature-column information used during model training.

Keeping the feature structure is important because prediction data must be transformed into the same feature structure that the model saw during training.

21. MODEL_METRICS.JOBLIB

File:

backend/ml/artifacts/model_metrics.joblib

This file stores model evaluation information.

The current training code saves metrics such as:

Precision
Recall
AUC

These metrics help evaluate the quality of the trained fraud-detection model.

22. DATABASE.PY

File:

backend/database.py

The database.py module contains database-related functionality.

The database layer is responsible for interacting with the project's local database and can be used to store and retrieve information required by the application.

Depending on the implementation, the database can store transaction information, risk assessments, analysis results, graph-related information, and other application data.

23. TRUST_GRAPH.DB

File:

backend/trust_graph.db

This is the project's local database file.

The database provides persistent storage for information used by the application.

Because database files can contain generated or local data, the database should normally be excluded from source control unless there is a specific reason to commit it.

24. GRAPH_ENGINE.PY

File:

backend/graph_engine.py

The graph engine is one of the main concepts behind the Trust-Graph architecture.

Fraud is often not limited to one transaction. Suspicious activity can exist through relationships between customers, devices, IP addresses, cards, sellers, email addresses, and delivery locations.

The graph engine is intended to represent these relationships.

A simplified relationship can be represented as:

Customer
    |
    | uses
    ↓
Device
    |
    | used by
    ↓
Another Customer

Another example is:

Customer
    |
    | purchases from
    ↓
Seller
    |
    | receives
    ↓
Transaction

By analyzing these connections, the system can identify relationship-based evidence that may not be obvious from a single transaction.

25. LLM.PY

File:

backend/llm.py

The llm.py module is responsible for LLM-related functionality.

The Large Language Model can be used to generate natural-language explanations, summarize evidence, assist with investigation, and produce recommendations.

API keys or other sensitive credentials required by the LLM should be stored in environment variables and never hard-coded into source files.

26. MAIN.PY

File:

backend/main.py

The main.py file is the main backend entry point.

It connects the application's components and provides the main execution flow.

Conceptually, a request or transaction can move through:

Transaction Input
    ↓
Machine-Learning Prediction
    ↓
Risk Scoring
    ↓
Graph Analysis
    ↓
AI Explanation
    ↓
Remediation Recommendation
    ↓
Self Check
    ↓
Final Response

The exact execution flow depends on the implementation inside main.py.

27. RISK_SCORER.PY

File:

backend/risk_scorer.py

The risk scorer is responsible for converting different risk signals into an overall risk assessment.

Possible inputs include:

Machine-learning probability.
Transaction characteristics.
Graph relationships.
Historical signals.
Rule-based signals.
Other available evidence.

The overall risk can then be categorized into levels such as:

LOW
MEDIUM
HIGH
CRITICAL

The exact thresholds and scoring formula depend on the implementation.

28. ENVIRONMENT FILE

File:

backend/.env

The .env file stores environment variables and sensitive configuration values.

Examples of values that may be stored in .env include:

Kaggle credentials.
LLM API keys.
Database configuration.
Other private configuration.

The .env file must never be committed to GitHub.

A safe practice is to create an example environment file such as:

.env.example

The example file can show the variable names without containing real credentials.

29. VIRTUAL ENVIRONMENT

Folder:

venv/

The venv folder is the Python virtual environment for the project.

It contains the Python interpreter and installed dependencies required by Trust-Graph.

The virtual environment should not be uploaded to GitHub.

Instead, dependencies should be stored in requirements.txt so another developer can recreate the environment.

30. GITIGNORE

File:

.gitignore

The .gitignore file prevents files that should not be committed from being uploaded to GitHub.

Recommended entries include:

.env
venv/
.venv/
__pycache__/
*.pyc
*.db
*.joblib
large raw dataset files

Large Kaggle datasets should generally be downloaded separately rather than committed into the Git repository.

31. KAGGLE DATASET

Trust-Graph uses the IEEE-CIS Fraud Detection dataset from Kaggle for machine-learning development.

The competition identifier is:

ieee-fraud-detection

The dataset contains transaction and identity information for fraud-detection research and model development.

The dataset is large, so the project should avoid unnecessarily duplicating the raw files.

32. KAGGLE ACCOUNT REQUIREMENT

A Kaggle account is required to access the competition dataset through KaggleHub.

The user must be logged into Kaggle and must have permission to access the competition.

Most importantly, authentication and competition permission are separate.

A valid Kaggle API token proves that the API credentials are valid.

It does not automatically mean that the account has accepted the rules for every Kaggle competition.

33. KAGGLE API LOGIN

The project uses KaggleHub for authentication and dataset downloading.

First, install KaggleHub inside the project's virtual environment:

pip install kagglehub

Then open Python:

python

Import KaggleHub:

import kagglehub

Start authentication:

kagglehub.login()

KaggleHub asks for the Kaggle API token.

The token is entered interactively and is not displayed on screen.

If the authentication is successful, the terminal displays a message similar to:

Kaggle credentials set.
Kaggle credentials successfully validated.

This confirms that the API credentials were successfully authenticated.

34. WHERE TO GET THE KAGGLE API TOKEN

Log into the Kaggle account that will be used for the project.

Open the Kaggle account settings and create or obtain an API token using the available API credential option.

The token must belong to the same Kaggle account that has access to the IEEE-CIS Fraud Detection competition.

Do not share the API token publicly.

Do not put the API token directly inside Python source code.

Do not commit the token to GitHub.

35. KAGGLE COMPETITION RULES

Before downloading the IEEE-CIS Fraud Detection competition files, open the competition page in a browser.

Competition page:

https://www.kaggle.com/competitions/ieee-fraud-detection

Rules page:

https://www.kaggle.com/competitions/ieee-fraud-detection/rules

Log into the correct Kaggle account.

If required, click Join Competition or Register.

Open the Rules section.

Accept the competition rules.

The account that accepts the rules must be the same Kaggle account associated with the API credentials being used by KaggleHub.

36. KAGGLE DATASET DOWNLOAD USING KAGGLEHUB

After successful authentication and competition access, the dataset can be downloaded using:

import kagglehub

path = kagglehub.competition_download("ieee-fraud-detection")

print("Path to competition files:", path)

KaggleHub downloads the competition files and returns the local path where they are stored.

37. DOWNLOADING TO A SPECIFIC DIRECTORY

If the project needs the files to be downloaded into a specific directory, use the output_dir option:

import kagglehub

path = kagglehub.competition_download(
    "ieee-fraud-detection",
    output_dir="./data"
)

print("Dataset downloaded to:", path)

The output directory should match the location expected by the project's data-loading code.

38. IMPORTANT KAGGLE 403 ERROR

If KaggleHub returns:

403 Client Error: Forbidden

and the message says:

You don't have permission to access resource at URL:
https://kaggle.com/competitions/ieee-fraud-detection

this usually means that the account is authenticated but does not have permission to download that competition's files.

The most common reason is that the competition rules have not been accepted for that Kaggle account.

The authentication process and competition access process are separate.

For example:

Kaggle API Token
        ↓
Authentication
        ↓
Credentials Validated
        ↓
Competition Permission
        ↓
Rules Accepted
        ↓
Dataset Download

If authentication succeeds but competition permission fails, Kaggle returns a 403 error.

39. IF KAGGLE CREDENTIALS ARE VALIDATED BUT DOWNLOAD STILL FAILS

If the terminal says:

Kaggle credentials successfully validated.

but the competition download still returns 403, do not immediately assume that the API token is invalid.

Instead check:

1. Is the browser logged into the same Kaggle account?
2. Has that account joined the competition?
3. Has that account accepted the competition rules?
4. Is the API token generated from that same account?
5. Does the account have access to the competition?

After verifying these points, run the download command again.

40. WHY PATH IS NOT DEFINED AFTER A FAILED DOWNLOAD

If this command fails:

path = kagglehub.competition_download("ieee-fraud-detection")

then the variable path is never created.

Therefore, running:

print(path)

will result in:

NameError: name 'path' is not defined

This is not a second Kaggle problem.

It is simply a consequence of the first command failing.

The correct order is:

First make the download succeed.

Then print the path.

For example:

path = kagglehub.competition_download("ieee-fraud-detection")
print(path)

41. DATA PREPARATION FLOW

After obtaining the dataset, the training script prepares the data.

The general preparation process is:

Raw Transaction Dataset
        +
Raw Identity Dataset
        ↓
Load into Pandas
        ↓
Merge using the appropriate transaction identifier
        ↓
Inspect columns
        ↓
Identify categorical features
        ↓
Encode categorical features
        ↓
Prepare model features
        ↓
Train/Test preparation
        ↓
XGBoost Training

The exact preprocessing implementation is defined inside train_model.py.

42. CATEGORICAL ENCODING

The training output currently shows:

Encoding 31 categorical columns...

Categorical columns contain non-numeric values such as categories or identifiers.

Machine-learning models require numerical representations for these features, so the training pipeline converts the required categorical information into a model-compatible format.

The same feature transformation logic must be applied when making predictions on new data.

43. MEMORY CONSIDERATIONS

The IEEE-CIS dataset is large.

After merging transaction and identity information, the dataset contains hundreds of columns.

The current training output shows:

Merged shape: (590540, 434)

and:

Final memory usage: 1.02 GB

The actual memory requirement of the complete process can be significantly higher than the final dataframe size because Pandas, NumPy, XGBoost, temporary arrays, copies, and model-training structures can all consume additional memory.

If the system produces an ArrayMemoryError, the computer may not have enough available RAM for the current operation.

44. MEMORY ERROR TROUBLESHOOTING

If an error such as:

numpy.core._exceptions._ArrayMemoryError

appears, possible solutions include:

Reduce unnecessary dataframe copies.
Use smaller numeric data types where appropriate.
Avoid unnecessary conversion of the complete dataframe into NumPy arrays.
Process large datasets in chunks when possible.
Reduce the number of features.
Use a smaller subset while developing.
Close unnecessary applications.
Use a machine with more available RAM.

Development training can initially be performed on a smaller dataset, while final training can use the complete dataset.

45. MODEL TRAINING COMMAND

After the dataset is correctly configured, activate the virtual environment.

On Windows PowerShell:

venv\Scripts\activate

Navigate to the backend directory:

cd backend

Run the training script:

python ml/train_model.py

The model-training process should then load the configured datasets and train the XGBoost model.

46. MODEL EVALUATION

The training script evaluates the model after training.

The current implementation saves metrics including:

Precision
Recall
AUC

Precision measures how many transactions predicted as fraud are actually fraud.

Recall measures how many actual fraudulent transactions are successfully detected.

AUC measures the model's ability to distinguish between fraudulent and legitimate transactions across classification thresholds.

For fraud detection, recall can be particularly important because missing fraudulent transactions can have significant consequences. However, precision is also important because too many false positives can create unnecessary investigation work.

47. MODEL PERSISTENCE

The trained model is saved using Joblib.

This allows the application to load the trained model later without retraining it every time.

The artifacts are stored in:

backend/ml/artifacts/

The expected artifacts include:

fraud_model.joblib
feature_columns.joblib
model_metrics.joblib

48. PREDICTION FLOW

Once the model has been trained, the backend can use the stored model for prediction.

The conceptual prediction flow is:

New Transaction
        ↓
Feature Preparation
        ↓
Apply Same Feature Structure
        ↓
Load fraud_model.joblib
        ↓
Generate Prediction / Probability
        ↓
Risk Scoring
        ↓
Graph Analysis
        ↓
AI Agents
        ↓
Final Result

49. RISK ANALYSIS

The machine-learning prediction is one part of the overall Trust-Graph analysis.

The system can combine:

Machine-learning risk.
Transaction-level signals.
Relationship-based signals.
Historical evidence.
Rule-based signals.
Other application-specific evidence.

This produces an overall risk assessment.

50. GRAPH-BASED TRUST ANALYSIS

The central idea of Trust-Graph is that fraud can involve networks of connected entities.

For example:

Customer A
    ↓
Device X
    ↓
Customer B
    ↓
Seller Z
    ↓
Multiple suspicious transactions

A single transaction may appear normal when considered alone.

However, if the same device, IP address, seller, email address, card, or delivery address is repeatedly associated with suspicious activity, the relationship itself becomes an important signal.

The graph engine is intended to capture this type of relationship.

51. EXPLAINABLE FRAUD DETECTION

Trust-Graph is designed to make fraud predictions more understandable.

Instead of returning only:

Fraud Probability = 0.87

the system can provide:

Risk Level: HIGH

Evidence:
The machine-learning model produced a high fraud probability.
The transaction has suspicious characteristics.
Connected entities may have previous relationships with other transactions.
Additional evidence was identified by graph analysis.

Recommendation:
Review the transaction and perform additional verification.

The exact output depends on the implemented agents and available evidence.

52. AI AGENT FLOW

The AI agents work as specialized stages.

The Explainer Agent answers:

"Why is this transaction risky?"

The Remediation Agent answers:

"What should be done next?"

The Self-Check Agent answers:

"Is the generated analysis consistent and supported by the available evidence?"

This creates a multi-stage reasoning pipeline:

Risk Detection
        ↓
Explanation
        ↓
Recommendation
        ↓
Validation
        ↓
Final Output

53. RUNNING THE BACKEND

After the environment, dataset, and model artifacts are ready, the backend can be started through the project's main entry point.

From the backend directory:

python main.py

The exact command may differ if the final backend is exposed through a framework such as FastAPI or Flask.

The main.py file is the source of truth for the current application startup process.

54. DEVELOPMENT WORKFLOW

A recommended development workflow is:

Step 1:
Create and activate the Python virtual environment.

Step 2:
Install the required dependencies.

Step 3:
Configure the .env file.

Step 4:
Authenticate with Kaggle.

Step 5:
Accept the IEEE-CIS Fraud Detection competition rules.

Step 6:
Download the dataset.

Step 7:
Place or configure the dataset according to train_model.py.

Step 8:
Run train_model.py.

Step 9:
Verify the generated model artifacts.

Step 10:
Start the backend.

Step 11:
Test transaction analysis.

Step 12:
Verify the risk score, graph evidence, explanation, remediation, and self-check output.

55. SECURITY AND PRIVACY

Never commit credentials, tokens, passwords, private API keys, or other secrets.

The following should normally be excluded from Git:

.env
venv/
.venv/
__pycache__/
*.pyc
*.db
*.joblib
large raw datasets

If a secret is accidentally committed to GitHub, it should be revoked or rotated immediately.

56. DATASET AND GITHUB MANAGEMENT

Large Kaggle datasets should not normally be uploaded to the Git repository.

Instead, the README should document how to obtain the dataset.

The recommended architecture is:

GitHub Repository
    ↓
Source Code
    ↓
README
    ↓
Requirements
    ↓
Configuration Example

Developer Machine
    ↓
Kaggle Authentication
    ↓
Dataset Download
    ↓
Local Dataset
    ↓
Model Training

This keeps the repository smaller and avoids distributing restricted or unnecessarily large raw datasets.

57. RECOMMENDED .GITIGNORE

A suitable .gitignore can contain:

.env
venv/
.venv/
__pycache__/
*.pyc
*.pyo
*.db
*.joblib
large raw datasets
external/

If a generated model is intentionally required for deployment, the artifact-management strategy can be changed accordingly.

58. TROUBLESHOOTING SUMMARY

Problem:
Kaggle credentials are validated but dataset download gives 403.

Solution:
Accept the competition rules and verify that the API token belongs to the same Kaggle account.

Problem:
path is not defined.

Solution:
The download command failed, so path was never assigned. Fix the original download error first.

Problem:
ArrayMemoryError.

Solution:
The current operation requires more memory than is available. Reduce memory usage, reduce the dataset during development, optimize dataframe operations, or use a machine with more RAM.

Problem:
ModuleNotFoundError.

Solution:
Activate the correct virtual environment and install the missing dependency.

Problem:
Model artifact is missing.

Solution:
Run train_model.py successfully and check backend/ml/artifacts/.

Problem:
Prediction feature mismatch.

Solution:
Ensure prediction preprocessing uses the same feature-column structure stored in feature_columns.joblib.

59. FUTURE DEVELOPMENT

Future versions of Trust-Graph can extend the current architecture with:

More advanced graph analytics.
NetworkX-based relationship analysis.
Neo4j or another graph database.
Graph embeddings.
Graph neural networks.
SHAP-based model explanations.
More advanced feature engineering.
Model hyperparameter optimization.
Real-time fraud scoring.
Real-time transaction monitoring.
Investigation dashboards.
User authentication.
REST APIs.
Cloud deployment.
Docker support.
Automated testing.
CI/CD pipelines.
Model monitoring.
Feedback-driven model improvement.
More advanced multi-agent coordination.

60. PROJECT VISION

The long-term vision of Trust-Graph is to transform fraud detection from a simple prediction task into a complete investigation and decision-support system.

The system should eventually be able to take a transaction, identify its risk, examine its surrounding entity network, collect supporting evidence, explain the reasoning, recommend an action, and validate the final result.

The target workflow is:

TRANSACTION
    ↓
DETECTION
    ↓
RISK SCORE
    ↓
GRAPH RELATIONSHIPS
    ↓
EVIDENCE
    ↓
EXPLANATION
    ↓
REMEDIATION
    ↓
SELF-CHECK
    ↓
TRUSTWORTHY DECISION

61. TECHNOLOGY STACK

Programming Language:
Python

Data Processing:
Pandas
NumPy

Machine Learning:
XGBoost
Scikit-learn

Model Persistence:
Joblib

Dataset Source:
Kaggle

Dataset Access:
KaggleHub

Database:
SQLite

AI / LLM:
LLM integration through llm.py

Backend:
Python backend application

Version Control:
Git and GitHub

Environment:
Python virtual environment

62. QUICK START

The shortest setup flow is:

Create virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Install KaggleHub if required:

pip install kagglehub

Authenticate:

python

import kagglehub
kagglehub.login()

Accept the IEEE-CIS Fraud Detection competition rules in the Kaggle browser.

Download:

path = kagglehub.competition_download("ieee-fraud-detection")

Train:

python ml/train_model.py

Verify:

backend/ml/artifacts/

Start the backend:

python main.py

63. IMPORTANT NOTE ABOUT THE CURRENT PROJECT

The README describes the architecture and intended responsibilities of the current Trust-Graph project. Individual implementation details may evolve as development continues.

The actual behavior of each component is determined by its current source code.

The most important source files for understanding the implementation are:

backend/main.py
backend/risk_scorer.py
backend/graph_engine.py
backend/database.py
backend/llm.py
backend/ml/train_model.py
backend/agents/explainer_agent.py
backend/agents/remediation_agent.py
backend/agents/selfcheck_agent.py

64. FINAL SUMMARY

Trust-Graph is a fraud-detection and trust-analysis system that combines machine learning with graph intelligence and AI agents.

The project starts with the IEEE-CIS Fraud Detection dataset. KaggleHub is used to authenticate with Kaggle and download the competition data after the required competition rules have been accepted.

The training pipeline loads transaction and identity data, merges the datasets, encodes categorical features, trains an XGBoost model, evaluates the model using precision, recall, and AUC, and stores the trained model and supporting artifacts.

The backend then provides the foundation for risk scoring, graph analysis, database operations, LLM-based reasoning, explanation, remediation, and self-checking.

The core idea of Trust-Graph is:

DO NOT ONLY ASK WHETHER A TRANSACTION IS FRAUDULENT.

ASK WHY IT IS RISKY, WHAT IT IS CONNECTED TO, WHAT EVIDENCE SUPPORTS THE RISK, AND WHAT SHOULD BE DONE NEXT.

This relationship-aware and explainable approach is the foundation of the Trust-Graph system.

Our fraud detection blends three independent signals — a classical graph algorithm that finds collusion rings, rule-based checks for known fraud patterns, and a trained ML model — into one explainable risk score per seller. Only borderline/high cases get expensive LLM reasoning; the scoring itself is fast and cheap.

transactions.csv + deliveries.csv
            │
            ├──────────────────────┬──────────────────────┐
            ▼                      ▼                       ▼
    graph_engine.py         risk_scorer.py            ml_scorer.py
  graph_anomaly_scores()   compute_rule_signals()   compute_ml_signals()
            │                      │                       │
   graph_risk_score          return_rate,            ml_fraud_score
   (0-1 per seller)       missing_proof_rate               │
            │                      │                       │
            └──────────────┬───────┴───────────────────────┘
                            ▼
                  risk_scorer.py: compute_final_risk()
                            │
              final_risk_score = 0.45×graph + 0.25×ML
                            + 0.15×return_rate + 0.15×missing_proof
                            │
                            ▼
                   classify_tier() → no_action / soft_intervention
                                      / hard_action_candidate
                            │
                            ▼
              needs_agent_review = score >= 0.4
              (ONLY these go on to cost an LLM call)

              