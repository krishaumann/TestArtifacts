# TestArtifacts
Generates Visual Test Model from Exported Tests (CSV and Feature)

ActiveLearningTestClassifier: 
- This script implements an active-learning workflow for classifying software
test cases into functional areas. It is a self-contained, well-documented,
and maintainable rewrite of the provided implementation.

GenerateTestModel:
- This script loads consolidated test paths from CSV exports,
builds a directed graph of functional area flows, and provides
an interactive GUI for filtering and visualizing the model.

ImportPipelneController:
- This script ingests test cases (CSV or Gherkin), validates input,
clusters test cases semantically, identifies functional areas,
builds representative path models, and exports enriched data.

TestAmbiquityDetector
- This script loads CSV files containing test cases, validates their structure,
and analyzes them for ambiguity using NLP and clustering techniques.

Pre-requisites
- CSV folder created with a list of test cases to be used.  See Sample
- Features folder created with a list of test cases to be used.  See Sample

