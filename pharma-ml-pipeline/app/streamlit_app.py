# File: app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import tensorflow as tf
from transformers import AutoTokenizer
import mlflow
from mlflow.tracking import MlflowClient
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import base64
from io import BytesIO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import KaggleDataLoader
from src.data.preprocessing import MolecularFeatureExtractor
from src.models.transformer_model import PharmaTransformerModel

# Set page configuration
st.set_page_config(
    page_title="Pharma ML Pipeline",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar configuration
st.sidebar.title("Pharma ML Pipeline")
st.sidebar.image("https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/_static/MLflow-logo-final-black.png", width=200)

# Configure MLflow
@st.cache_resource
def setup_mlflow():
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    return MlflowClient()

mlflow_client = setup_mlflow()

# App pages
pages = [
    "Dashboard",
    "Data Explorer",
    "Model Training",
    "Model Evaluation",
    "Prediction",
    "Model Registry"
]

page = st.sidebar.selectbox("Navigation", pages)

# Helper functions
def render_mol_image(smiles, width=400, height=200):
    """Render molecule image from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(width, height))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_str}" width="{width}" height="{height}">'
    return "Invalid SMILES"

def load_experiment_runs(experiment_name):
    """Load runs for a specific experiment"""
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        return runs
    return []

def load_registered_models():
    """Load all registered models"""
    return mlflow_client.search_registered_models()

# Dashboard Page
if page == "Dashboard":
    st.title("Pharmaceutical Machine Learning Pipeline")
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    try:
        # Get number of experiments
        experiments = mlflow_client.search_experiments()
        
        # Get number of models
        registered_models = mlflow_client.search_registered_models()
        
        # Get number of runs
        total_runs = 0
        for exp in experiments:
            runs = mlflow_client.search_runs(experiment_ids=[exp.experiment_id])
            total_runs += len(runs)
            
        col1.metric("Experiments", len(experiments))
        col2.metric("Registered Models", len(registered_models))
        col3.metric("Total Runs", total_runs)
    except Exception as e:
        st.error(f"Failed to fetch MLflow metrics: {str(e)}")
        st.info("Make sure MLflow server is running and accessible.")
    
    # Recent runs
    st.subheader("Recent Model Training Runs")
    
    try:
        # Get all experiments
        experiments = mlflow_client.search_experiments()
        
        all_runs = []
        for exp in experiments:
            runs = mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=5
            )
            for run in runs:
                all_runs.append({
                    "Run ID": run.info.run_id,
                    "Experiment": exp.name,
                    "Status": run.info.status,
                    "Start Time": pd.to_datetime(run.info.start_time, unit='ms'),
                    "Duration": (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else "Running",
                    "Metrics": ", ".join([f"{k}: {v:.4f}" for k, v in run.data.metrics.items() if k.startswith("eval_")])
                })
                
        if all_runs:
            runs_df = pd.DataFrame(all_runs)
            st.dataframe(runs_df)
        else:
            st.info("No runs found. Train some models to see them here.")
    except Exception as e:
        st.error(f"Failed to fetch MLflow runs: {str(e)}")
    
    # Model performance comparison
    st.subheader("Model Performance Comparison")
    
    try:
        # Get all runs with evaluation metrics
        experiments = mlflow_client.search_experiments()
        
        eval_metrics = []
        for exp in experiments:
            runs = mlflow_client.search_runs(experiment_ids=[exp.experiment_id])
            for run in runs:
                metrics = {k.replace("eval_", ""): v for k, v in run.data.metrics.items() if k.startswith("eval_")}
                if metrics:
                    metrics["Run ID"] = run.info.run_id
                    metrics["Model Type"] = run.data.params.get("model_type", "Unknown")
                    metrics["Experiment"] = exp.name
                    eval_metrics.append(metrics)
                    
        if eval_metrics:
            metrics_df = pd.DataFrame(eval_metrics)
            
            # Plot metrics
            if "accuracy" in metrics_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=metrics_df, x="Model Type", y="accuracy", hue="Experiment", ax=ax)
                plt.title("Model Accuracy Comparison")
                plt.ylabel("Accuracy")
                plt.xlabel("Model Type")
                plt.tight_layout()
                st.pyplot(fig)
            elif "mae" in metrics_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=metrics_df, x="Model Type", y="mae", hue="Experiment", ax=ax)
                plt.title("Model Mean Absolute Error Comparison")
                plt.ylabel("MAE (lower is better)")
                plt.xlabel("Model Type")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No evaluation metrics found. Train and evaluate models to see comparisons.")
    except Exception as e:
        st.error(f"Failed to create performance comparison: {str(e)}")

# Data Explorer Page
elif page == "Data Explorer":
    st.title("Pharmaceutical Data Explorer")
    
    # Dataset selection
    dataset_source = st.radio(
        "Data Source",
        ["Upload CSV", "Kaggle Dataset", "Sample Dataset"]
    )
    
    if dataset_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column selection for analysis
            if "SMILES" in df.columns or any("smiles" in col.lower() for col in df.columns):
                smiles_cols = [col for col in df.columns if "SMILES" in col or "smiles" in col.lower()]
                smiles_col = st.selectbox("Select SMILES column", smiles_cols)
                
                # Molecule visualization
                st.subheader("Molecule Visualization")
                sample_idx = st.slider("Select molecule index", 0, min(len(df)-1, 100), 0)
                smiles = df.iloc[sample_idx][smiles_col]
                st.markdown(render_mol_image(smiles, width=600, height=300), unsafe_allow_html=True)
                
                # Molecular descriptor calculation
                if st.button("Calculate Molecular Descriptors"):
                    with st.spinner("Calculating descriptors..."):
                        extractor = MolecularFeatureExtractor()
                        descriptors_df = extractor.calculate_descriptors(df[smiles_col].head(100))
                        st.dataframe(descriptors_df.head())
                        
                        # Show correlation heatmap of top descriptors
                        st.subheader("Descriptor Correlation Heatmap")
                        corr = descriptors_df.select_dtypes(include=[np.number]).corr()
                        
                        # Get top correlated features
                        top_corr = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
                        top_features = list(set([i[0] for i in top_corr.head(50).index if i[0] != i[1]] + 
                                              [i[1] for i in top_corr.head(50).index if i[0] != i[1]]))
                        
                        fig, ax = plt.subplots(figsize=(12, 10))
                        sns.heatmap(corr.loc[top_features, top_features], annot=False, cmap="coolwarm", ax=ax)
                        plt.tight_layout()
                        st.pyplot(fig)
            
            # Statistical analysis
            st.subheader("Statistical Analysis")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Select column for analysis", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Summary Statistics")
                    st.write(df[selected_col].describe())
                
                with col2:
                    # Distribution plot
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
                    plt.title(f"Distribution of {selected_col}")
                    plt.tight_layout()
                    st.pyplot(fig)
    
    elif dataset_source == "Kaggle Dataset":
        st.warning("This feature requires Kaggle API credentials to be configured")
        
        kaggle_dataset = st.text_input("Enter Kaggle dataset name (e.g., 'user/dataset-name')")
        
        if kaggle_dataset:
            try:
                with st.spinner("Downloading dataset..."):
                    loader = KaggleDataLoader()
                    dataset_path = loader.download_dataset(kaggle_dataset)
                    
                    # Find CSV files
                    import glob
                    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
                    
                    if csv_files:
                        selected_csv = st.selectbox("Select CSV file", csv_files)
                        df = pd.read_csv(selected_csv)
                        
                        # Show data preview
                        st.subheader("Data Preview")
                        st.dataframe(df.head())
                    else:
                        st.error("No CSV files found in the dataset")
            except Exception as e:
                st.error(f"Error downloading dataset: {str(e)}")
                st.info("Make sure you have configured the Kaggle API credentials correctly")
    
    else:  # Sample Dataset
        st.info("Loading sample pharmaceutical dataset")
        
        # Create a sample dataset
        sample_data = {
            "SMILES": [
                "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
                "CCC1=CC=CC=C1NC(=O)C",      # Acetanilide
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
                "CC(C)NCC(O)COC1=CC=C(C=C1)CCOC",  # Metoprolol
                "CC1=CN=C(C=N1)CN2C3CC4C(C2C5=C3C=CC(=C5)O)CCN4C"  # Quinine
            ],
            "MolWeight": [180.16, 135.17, 194.19, 267.36, 324.42],
            "Activity": [0.78, 0.42, 0.63, 0.85, 0.91],
            "IC50_nM": [12, 45, 28, 6, 3],
            "LogP": [1.19, 1.08, -0.07, 1.88, 3.44]
        }
        
        df = pd.DataFrame(sample_data)
        st.dataframe(df)
        
        # Molecule visualization
        st.subheader("Molecule Visualization")
        sample_idx = st.slider("Select molecule index", 0, len(df)-1, 0)
        smiles = df.iloc[sample_idx]["SMILES"]
        st.markdown(render_mol_image(smiles, width=600, height=300), unsafe_allow_html=True)
        
        # Activity vs Properties plot
        st.subheader("Structure-Activity Relationship")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df["LogP"], df["Activity"], c=df["IC50_nM"], s=df["MolWeight"]/3, cmap="viridis")
        plt.xlabel("LogP")
        plt.ylabel("Activity")
        plt.colorbar(scatter, label="IC50 (nM)")
        plt.title("Activity vs LogP colored by IC50")
        st.pyplot(fig)

# Model Training Page
elif page == "Model Training":
    st.title("Train Pharmaceutical Models")
    
    # Model type selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Transformer-based NLP Model", "CNN for Molecular Fingerprints"]
    )
    
    # Common parameters
    dataset_source = st.radio(
        "Data Source",
        ["Upload CSV", "Kaggle Dataset", "Sample Dataset"]
    )
    
    if dataset_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            st.dataframe(df.head())
    
    elif dataset_source == "Kaggle Dataset":
        kaggle_dataset = st.text_input("Enter Kaggle dataset name (e.g., 'user/dataset-name')")
    
    else:  # Sample Dataset
        st.info("Using sample pharmaceutical dataset for training")
    
    # Model-specific parameters
    if model_type == "Transformer-based NLP Model":
        st.subheader("Transformer Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pretrained_model = st.selectbox(
                "Pretrained Model",
                ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "albert-base-v2"]
            )
            task_type = st.selectbox(
                "Task Type",
                ["Binary Classification", "Multi-class Classification", "Regression"]
            )
            text_col = st.text_input("Text Column Name", "Description")
            target_col = st.text_input("Target Column Name", "Activity")
        
        with col2:
            epochs = st.slider("Training Epochs", 1, 50, 5)
            batch_size = st.slider("Batch Size", 4, 64, 16)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
                format_func=lambda x: f"{x:.0e}"
            )
            max_length = st.slider("Maximum Sequence Length", 64, 512, 128)
        
        # Training button
        if st.button("Train Transformer Model"):
            st.info("Training not implemented in this demo interface")
            st.success("Training would be started with the following parameters:")
            
            params = {
                "Model Type": "Transformer",
                "Pretrained Model": pretrained_model,
                "Task Type": task_type,
                "Text Column": text_col,
                "Target Column": target_col,
                "Epochs": epochs,
                "Batch Size": batch_size,
                "Learning Rate": learning_rate,
                "Max Length": max_length
            }
            
            st.json(params)
            
            # Display expected pipeline code
            st.code(f"""
# Example code that would run:
from src.training.trainer import PharmaModelTrainer

trainer = PharmaModelTrainer(experiment_name="pharma-model")

# Determine num_classes based on task type
num_classes = None  # for regression
if "{task_type}" == "Binary Classification":
    num_classes = 2
elif "{task_type}" == "Multi-class Classification":
    num_classes = len(df["{target_col}"].unique())

# Train the model
model = trainer.train_transformer_model(
    dataset_name="{kaggle_dataset if dataset_source == 'Kaggle Dataset' else 'local_dataset'}",
    text_col="{text_col}",
    target_col="{target_col}",
    pretrained_model="{pretrained_model}",
    num_classes=num_classes,
    epochs={epochs},
    batch_size={batch_size},
    learning_rate={learning_rate},
    max_length={max_length}
)
            """, language="python")
    
    else:  # CNN for Molecular Fingerprints
        st.subheader("CNN Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            task_type = st.selectbox(
                "Task Type",
                ["Binary Classification", "Multi-class Classification", "Regression"]
            )
            smiles_col = st.text_input("SMILES Column Name", "SMILES")
            target_col = st.text_input("Target Column Name", "Activity")
            fp_type = st.selectbox(
                "Fingerprint Type",
                ["Morgan (ECFP)", "RDKit", "MACCS Keys"]
            )
        
        with col2:
            epochs = st.slider("Training Epochs", 10, 200, 50)
            batch_size = st.slider("Batch Size", 16, 256, 32)
            fp_radius = st.slider("Fingerprint Radius", 2, 6, 3)
            fp_bits = st.select_slider(
                "Fingerprint Size",
                options=[1024, 2048, 4096]
            )
        
        # Training button
        if st.button("Train CNN Model"):
            st.info("Training not implemented in this demo interface")
            st.success("Training would be started with the following parameters:")
            
            params = {
                "Model Type": "CNN for Molecular Fingerprints",
                "Task Type": task_type,
                "SMILES Column": smiles_col,
                "Target Column": target_col,
                "Fingerprint Type": fp_type,
                "Fingerprint Radius": fp_radius,
                "Fingerprint Size": fp_bits,
                "Epochs": epochs,
                "Batch Size": batch_size
            }
            
            st.json(params)
            
            # Display expected pipeline code
            st.code(f"""
# Example code that would run:
from src.training.trainer import PharmaModelTrainer

trainer = PharmaModelTrainer(experiment_name="pharma-model")

# Determine num_classes based on task type
num_classes = None  # for regression
if "{task_type}" == "Binary Classification":
    num_classes = 2
elif "{task_type}" == "Multi-class Classification":
    num_classes = len(df["{target_col}"].unique())

# Train the model
model = trainer.train_cnn_model(
    dataset_name="{kaggle_dataset if dataset_source == 'Kaggle Dataset' else 'local_dataset'}",
    smiles_col="{smiles_col}",
    target_col="{target_col}",
    num_classes=num_classes,
    epochs={epochs},
    batch_size={batch_size}
)
            """, language="python")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    
    # Select experiment
    experiments = mlflow_client.search_experiments()
    experiment_names = [exp.name for exp in experiments]
    
    if experiment_names:
        selected_experiment = st.selectbox("Select Experiment", experiment_names)
        
        # Get runs for the selected experiment
        runs = load_experiment_runs(selected_experiment)
        
        if runs:
            run_infos = [(run.info.run_id, run.data.tags.get("mlflow.runName", run.info.run_id)) 
                         for run in runs]
            
            selected_run_id, selected_run_name = run_infos[st.selectbox(
                "Select Run",
                range(len(run_infos)),
                format_func=lambda i: f"{run_infos[i][1]} ({run_infos[i][0]})"
            )]
            
            # Get the selected run
            selected_run = next((run for run in runs if run.info.run_id == selected_run_id), None)
            
            if selected_run:
                # Display run info
                st.subheader("Run Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Start Time:", pd.to_datetime(selected_run.info.start_time, unit='ms'))
                    st.write("Status:", selected_run.info.status)
                    
                    # Calculate duration if run is completed
                    if selected_run.info.end_time:
                        duration = (selected_run.info.end_time - selected_run.info.start_time) / 1000
                        st.write("Duration:", f"{duration:.2f} seconds")
                
                with col2:
                    # Display parameters
                    st.write("Parameters:")
                    for param, value in selected_run.data.params.items():
                        st.write(f"- {param}: {value}")
                
                # Display metrics
                st.subheader("Metrics")
                
                # Separate training and evaluation metrics
                train_metrics = {k: v for k, v in selected_run.data.metrics.items() 
                                if not k.startswith("eval_")}
                eval_metrics = {k.replace("eval_", ""): v for k, v in selected_run.data.metrics.items() 
                               if k.startswith("eval_")}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Training Metrics:")
                    for metric, value in train_metrics.items():
                        st.write(f"- {metric}: {value:.4f}")
                
                with col2:
                    st.write("Evaluation Metrics:")
                    for metric, value in eval_metrics.items():
                        st.write(f"- {metric}: {value:.4f}")
                
                # Plot metrics if available
                metrics_to_plot = [metric for metric in train_metrics.keys() 
                                  if any(step_metric.startswith(f"{metric}_step_")
                                        for step_metric in train_metrics.keys())]
                
                if metrics_to_plot:
                    st.subheader("Training Progress")
                    
                    metric_to_plot = st.selectbox("Select Metric to Plot", metrics_to_plot)
                    
                    # Get step and value for the selected metric
                    steps = []
                    values = []
                    
                    for step_metric, value in train_metrics.items():
                        if step_metric.startswith(f"{metric_to_plot}_step_"):
                            step = int(step_metric.split("_")[-1])
                            steps.append(step)
                            values.append(value)
                    
                    if steps and values:
                        # Create plot
                        fig, ax = plt.subplots()
                        ax.plot(steps, values)
                        ax.set_xlabel("Step")
                        ax.set_ylabel(metric_to_plot)
                        ax.set_title(f"{metric_to_plot} during Training")
                        st.pyplot(fig)
                
                # Display artifacts
                st.subheader("Artifacts")
                
                client = MlflowClient()
                artifacts = client.list_artifacts(selected_run_id)
                
                if artifacts:
                    for artifact in artifacts:
                        st.write(f"- {artifact.path} ({artifact.file_size} bytes)")
                        
                        # If it's a model, offer download
                        if artifact.path == "model":
                            st.download_button(
                                "Download Model",
                                data="Not implemented in demo",
                                file_name="model.h5",
                                mime="application/octet-stream"
                            )
                else:
                    st.info("No artifacts found for this run")
        else:
            st.info("No runs found for the selected experiment. Train some models first.")
    else:
        st.info("No experiments found. Start training models to create experiments.")

# Prediction Page
elif page == "Prediction":
    st.title("Model Prediction")
    
    # Select registered model
    registered_models = load_registered_models()
    
    if registered_models:
        model_names = [model.name for model in registered_models]
        selected_model_name = st.selectbox("Select Model", model_names)
        
        # Get latest version of the selected model
        client = MlflowClient()
        model_versions = client.get_latest_versions(selected_model_name, stages=["None", "Staging", "Production"])
        
        if model_versions:
            version_infos = [(v.version, v.current_stage) for v in model_versions]
            selected_version_idx = st.selectbox(
                "Select Version",
                range(len(version_infos)),
                format_func=lambda i: f"Version {version_infos[i][0]} ({version_infos[i][1]})"
            )
            
            selected_version = model_versions[selected_version_idx]
            
            # Display model info
            st.subheader("Model Information")
            st.write(f"Name: {selected_model_name}")
            st.write(f"Version: {selected_version.version}")
            st.write(f"Stage: {selected_version.current_stage}")
            st.write(f"Created: {pd.to_datetime(selected_version.creation_timestamp, unit='ms')}")
            
            # Determine input type based on model name
            is_text_model = "transformer" in selected_model_name.lower()
            is_mol_model = "cnn" in selected_model_name.lower() or "fingerprint" in selected_model_name.lower()
            
            # Input form
            st.subheader("Make Prediction")
            
            if is_text_model:
                text_input = st.text_area("Enter text for prediction", height=150)
                
                if st.button("Predict (Text)"):
                    if text_input:
                        st.info("Prediction functionality not implemented in demo")
                        st.success("Example code for prediction:")
                        
                        st.code(f"""
# Example prediction code:
import mlflow

# Load model as a PyFuncModel
model_uri = f"models:/{selected_model_name}/{selected_version.version}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Preprocess input
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
inputs = tokenizer(["{text_input}"], padding="max_length", truncation=True, max_length=512, return_tensors="tf")

# Make prediction
prediction = loaded_model.predict(inputs)
print(f"Prediction: {{prediction}}")
                        """, language="python")
            
            elif is_mol_model:
                smiles_input = st.text_input("Enter SMILES string for prediction", "CC(=O)OC1=CC=CC=C1C(=O)O")
                
                # Show the molecule
                if smiles_input:
                    st.markdown(render_mol_image(smiles_input, width=400, height=200), unsafe_allow_html=True)
                
                if st.button("Predict (Molecular)"):
                    if smiles_input:
                        st.info("Prediction functionality not implemented in demo")
                        st.success("Example code for prediction:")
                        
                        st.code(f"""
# Example prediction code:
import mlflow
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Load model as a PyFuncModel
model_uri = f"models:/{selected_model_name}/{selected_version.version}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Preprocess SMILES
mol = Chem.MolFromSmiles("{smiles_input}")
if mol:
    # Generate Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)
    
    # Make prediction
    prediction = loaded_model.predict(fp_array.reshape(1, -1))
    print(f"Prediction: {{prediction}}")
else:
    print("Invalid SMILES string")
                        """, language="python")
            
            else:
                st.warning("Unknown model type. Please provide appropriate input.")
                
                # Generic prediction
                json_input = st.text_area("Enter JSON input data", "{}")
                
                if st.button("Predict (Generic)"):
                    if json_input:
                        st.info("Prediction functionality not implemented in demo")
    else:
        st.info("No registered models found. Train and register models first.")

# Model Registry Page
elif page == "Model Registry":
    st.title("Model Registry")
    
    # List all registered models
    registered_models = load_registered_models()
    
    if registered_models:
        st.subheader("Registered Models")
        
        for model in registered_models:
            with st.expander(f"{model.name}"):
                st.write(f"Description: {model.description}")
                
                # Get model versions
                versions = client.search_model_versions(f"name='{model.name}'")
                
                # Create a table of versions
                version_data = []
                for version in versions:
                    version_data.append({
                        "Version": version.version,
                        "Stage": version.current_stage,
                        "Created": pd.to_datetime(version.creation_timestamp, unit='ms'),
                        "Last Updated": pd.to_datetime(version.last_updated_timestamp, unit='ms'),
                        "Run ID": version.run_id,
                        "Source": version.source
                    })
                
                if version_data:
                    version_df = pd.DataFrame(version_data)
                    st.dataframe(version_df)
                    
                    # Allow transitioning versions to different stages
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        version = st.selectbox(f"Version for {model.name}", 
                                              [v["Version"] for v in version_data],
                                              key=f"version_select_{model.name}")
                    
                    with col2:
                        stage = st.selectbox(f"Transition to Stage", 
                                            ["None", "Staging", "Production", "Archived"],
                                            key=f"stage_select_{model.name}")
                    
                    with col3:
                        if st.button("Transition", key=f"transition_button_{model.name}"):
                            st.info(f"Would transition version {version} to {stage} (not implemented in demo)")
                            
                            # Display code that would run
                            st.code(f"""
# Code that would run:
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="{model.name}",
    version="{version}",
    stage="{stage}"
)
                            """, language="python")
                else:
                    st.info("No versions found for this model")
    else:
        st.info("No registered models found. Train and register models first.")
