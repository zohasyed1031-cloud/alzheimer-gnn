from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Alzheimer GNN Analysis API",
    description="Research prototype for Alzheimer's classification using gene expression.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

selected_genes_df = pd.read_csv(
    MODEL_DIR / "selected_genes_1000.csv"
)

selected_genes_df.columns = (
    selected_genes_df.columns
    .astype(str)
    .str.strip()
)

if "Gene_Symbol" not in selected_genes_df.columns:
    raise RuntimeError(
        "selected_genes_1000.csv must contain a 'Gene_Symbol' column."
    )


selected_genes = (
    selected_genes_df["Gene_Symbol"]
    .astype(str)
    .str.strip()
    .tolist()
)


# Remove accidental duplicate selected genes
selected_genes = list(
    dict.fromkeys(selected_genes)
)


scaler = joblib.load(
    MODEL_DIR / "scaler.pkl"
)


edge_index = torch.load(
    MODEL_DIR / "edge_index.pt",
    map_location=device,
    weights_only=True
)


edge_weight = torch.load(
    MODEL_DIR / "edge_weight.pt",
    map_location=device,
    weights_only=True
)


edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)


# ============================================================
# MODEL
# ============================================================

class AlzheimerGCNWeighted(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(1, 64)

        self.conv2 = GCNConv(64, 64)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(128, 3)


    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        batch
    ):

        x = self.conv1(
            x,
            edge_index,
            edge_weight=edge_weight
        )

        x = F.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
            edge_weight=edge_weight
        )

        x = F.relu(x)

        mean_pool = global_mean_pool(
            x,
            batch
        )

        max_pool = global_max_pool(
            x,
            batch
        )

        x = torch.cat(
            [mean_pool, max_pool],
            dim=1
        )

        x = self.fc(x)

        return x


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = AlzheimerGCNWeighted().to(device)


model.load_state_dict(
    torch.load(
        MODEL_DIR / "weighted_improved_gcn.pth",
        map_location=device,
        weights_only=True
    )
)


model.eval()


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = {
    0: "AD",
    1: "CTL",
    2: "MCI"
}


CLASS_DESCRIPTIONS = {
    "AD": "Alzheimer's Disease",
    "CTL": "Control",
    "MCI": "Mild Cognitive Impairment"
}


# ============================================================
# BIOMARKERS
# ============================================================

BIOMARKER_COLUMNS = [
    "AB42",
    "AB40",
    "pTau181",
    "pTau217",
    "AB42_AB40_Ratio"
]


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status": "running",

        "model":
            "Weighted Improved GCN",

        "classes": [
            "AD",
            "CTL",
            "MCI"
        ],

        "selected_genes":
            len(selected_genes),

        "graph_edges":
            int(edge_index.shape[1]),

        "device":
            str(device)

    }


# ============================================================
# HELPER:
# CLEAN COLUMN NAMES
# ============================================================

def clean_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# HELPER:
# CREATE CASE-INSENSITIVE GENE MAP
# ============================================================

def create_gene_column_map(df):

    column_lookup = {}

    for column in df.columns:

        cleaned_column = str(
            column
        ).strip()

        key = cleaned_column.upper()

        if key not in column_lookup:

            column_lookup[key] = cleaned_column

    return column_lookup


# ============================================================
# HELPER:
# CHECK PATIENT GENES
# ============================================================

def check_patient_genes(df):

    column_lookup = create_gene_column_map(df)

    missing_genes = []

    gene_column_map = {}

    for gene in selected_genes:

        gene_clean = str(
            gene
        ).strip()

        gene_key = gene_clean.upper()

        if gene_key in column_lookup:

            gene_column_map[
                gene_clean
            ] = column_lookup[gene_key]

        else:

            missing_genes.append(
                gene_clean
            )

    return (
        missing_genes,
        gene_column_map
    )


# ============================================================
# HELPER:
# RUN GNN FOR ONE SAMPLE
# ============================================================

def run_gnn_prediction(sample_values):

    if len(sample_values) != len(selected_genes):

        raise ValueError(
            f"Expected {len(selected_genes)} gene values, "
            f"but received {len(sample_values)}."
        )


    sample_df = pd.DataFrame(
        [sample_values],
        columns=selected_genes
    )


    # SAME scaler used during training
    sample_scaled = scaler.transform(
        sample_df
    )


    # Convert to graph node features
    x = torch.tensor(
        sample_scaled[0],
        dtype=torch.float32,
        device=device
    ).view(-1, 1)


    # One graph = one batch
    batch = torch.zeros(
        x.size(0),
        dtype=torch.long,
        device=device
    )


    with torch.no_grad():

        output = model(
            x,
            edge_index,
            edge_weight,
            batch
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


        predicted_class = torch.argmax(
            probabilities
        ).item()


    prediction = CLASS_NAMES[
        predicted_class
    ]


    probability_values = {

        "AD": round(
            probabilities[0].item() * 100,
            2
        ),

        "CTL": round(
            probabilities[1].item() * 100,
            2
        ),

        "MCI": round(
            probabilities[2].item() * 100,
            2
        )

    }


    confidence = probability_values[
        prediction
    ]


    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probability_values

    }


# ============================================================
# HELPER:
# TOP GENE SIGNALS
# ============================================================

def get_top_gene_signals(sample_values):

    sample_df = pd.DataFrame(
        [sample_values],
        columns=selected_genes
    )


    scaled_values = scaler.transform(
        sample_df
    )[0]


    signal_df = pd.DataFrame({

        "gene":
            selected_genes,

        "signal":
            np.abs(scaled_values),

        "value":
            sample_values

    })


    signal_df = (
        signal_df
        .sort_values(
            "signal",
            ascending=False
        )
        .head(10)
    )


    return [

        {

            "gene":
                str(row["gene"]),

            "signal":
                round(
                    float(row["signal"]),
                    4
                ),

            "expression":
                round(
                    float(row["value"]),
                    4
                )

        }

        for _, row
        in signal_df.iterrows()

    ]


# ============================================================
# RESEARCH DATASET PREDICTION
#
# Expected:
#
# Gene_Symbol | GSM001 | GSM002 | ...
#
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()


        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )


        df = pd.read_csv(
            io.BytesIO(contents)
        )


        df = clean_columns(
            df
        )


        if "Gene_Symbol" not in df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Research dataset must contain "
                    "'Gene_Symbol' column."
                )
            )


        # Remove duplicate genes
        df = df.drop_duplicates(
            subset=["Gene_Symbol"]
        )


        df["Gene_Symbol"] = (
            df["Gene_Symbol"]
            .astype(str)
            .str.strip()
        )


        df = df.set_index(
            "Gene_Symbol"
        )


        if len(df.columns) == 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "CSV must contain at least "
                    "one sample column."
                )
            )


        # ----------------------------------------------------
        # CHECK REQUIRED GENES
        # ----------------------------------------------------

        missing_genes = [

            gene

            for gene
            in selected_genes

            if gene not in df.index

        ]


        if missing_genes:

            raise HTTPException(
                status_code=400,
                detail={

                    "message": (
                        "Uploaded dataset is missing "
                        "required genes."
                    ),

                    "missing_count":
                        len(missing_genes),

                    "required_genes":
                        len(selected_genes),

                    "example_missing_genes":
                        missing_genes[:30]

                }
            )


        expression = df.loc[
            selected_genes
        ]


        results = []


        for sample_id in expression.columns:

            sample = pd.to_numeric(
                expression[sample_id],
                errors="coerce"
            )


            if sample.isna().any():

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Non-numeric or missing values "
                        f"found in sample {sample_id}."
                    )
                )


            prediction_result = (
                run_gnn_prediction(
                    sample.values
                )
            )


            results.append({

                "sample_id":
                    str(sample_id),

                "prediction":
                    prediction_result[
                        "prediction"
                    ],

                "confidence":
                    prediction_result[
                        "confidence"
                    ],

                "probabilities":
                    prediction_result[
                        "probabilities"
                    ]

            })


        # ----------------------------------------------------
        # DATASET ANALYSIS
        # ----------------------------------------------------

        number_of_genes = len(
            df.index
        )

        number_of_samples = len(
            df.columns
        )


        numeric_df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )


        missing_values = int(
            numeric_df
            .isna()
            .sum()
            .sum()
        )


        mean_expression = float(
            numeric_df
            .mean()
            .mean()
        )


        gene_variances = numeric_df.var(
            axis=1
        )


        top_variable_genes_series = (
            gene_variances
            .sort_values(
                ascending=False
            )
            .head(10)
        )


        top_variable_genes = [

            {

                "gene":
                    str(gene),

                "variance":
                    round(
                        float(variance),
                        6
                    )

            }

            for gene, variance
            in top_variable_genes_series.items()

        ]


        # ----------------------------------------------------
        # EXPRESSION OVERVIEW
        # ----------------------------------------------------

        expression_overview = []


        if len(numeric_df.columns) > 0:

            first_sample = (
                numeric_df.iloc[:, 0]
            )


            overview_genes = (
                first_sample
                .dropna()
                .head(30)
            )


            expression_overview = [

                {

                    "gene":
                        str(gene),

                    "expression":
                        round(
                            float(value),
                            4
                        )

                }

                for gene, value
                in overview_genes.items()

            ]


        return {

            "status":
                "success",

            "dataset_type":
                "research",

            "sample_count":
                len(results),

            "number_of_genes":
                number_of_genes,

            "number_of_samples":
                number_of_samples,

            "missing_values":
                missing_values,

            "mean_expression":
                round(
                    mean_expression,
                    4
                ),

            "top_variable_genes":
                top_variable_genes,

            "expression_overview":
                expression_overview,

            "model":
                "Weighted Improved GCN",

            "selected_gene_count":
                len(selected_genes),

            "graph_edges":
                int(edge_index.shape[1]),

            "k_value":
                10,

            "results":
                results

        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# PATIENT PREDICTION
#
# Expected patient CSV:
#
# Patient_ID
# Sample_Type
# AB42
# AB40
# pTau181
# pTau217
# AB42_AB40_Ratio
# ABCA7
# ABCE1
# ...
#
# NO Gene_Symbol column is required.
#
# Diagnosis is OPTIONAL and ignored during prediction.
#
# ============================================================

@app.post("/patient-predict")
async def patient_predict(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()


        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Patient CSV is empty."
            )


        # ----------------------------------------------------
        # READ CSV
        # ----------------------------------------------------

        try:

            df = pd.read_csv(
                io.BytesIO(contents)
            )

        except Exception as e:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read the patient CSV. "
                    f"Error: {str(e)}"
                )
            )


        # Clean spaces from column names
        df = clean_columns(
            df
        )


        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        if len(df.columns) == 0:

            raise HTTPException(
                status_code=400,
                detail="Patient CSV contains no columns."
            )


        if len(df) == 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Patient CSV contains no "
                    "patient records."
                )
            )


        # ----------------------------------------------------
        # REQUIRED PATIENT COLUMNS
        # ----------------------------------------------------

        required_patient_columns = [
            "Patient_ID",
            "Sample_Type"
        ]


        missing_patient_columns = [

            column

            for column
            in required_patient_columns

            if column not in df.columns

        ]


        if missing_patient_columns:

            raise HTTPException(
                status_code=400,
                detail={

                    "message":
                        "Patient CSV is missing required columns.",

                    "missing_columns":
                        missing_patient_columns,

                    "required_columns":
                        required_patient_columns

                }
            )


        # ----------------------------------------------------
        # ONE PATIENT PER CSV
        # ----------------------------------------------------

        if len(df) > 1:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Please upload one patient record "
                    "at a time."
                )
            )


        patient = df.iloc[0]


        # ----------------------------------------------------
        # PATIENT ID
        # ----------------------------------------------------

        patient_id = str(
            patient["Patient_ID"]
        ).strip()


        if not patient_id:

            raise HTTPException(
                status_code=400,
                detail="Patient_ID cannot be empty."
            )


        # ----------------------------------------------------
        # SAMPLE TYPE
        # ----------------------------------------------------

        sample_type = str(
            patient["Sample_Type"]
        ).strip()


        if sample_type.lower() not in [
            "blood",
            "saliva"
        ]:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Sample_Type must be either "
                    "'Blood' or 'Saliva'."
                )
            )


        sample_type = (

            "Blood"

            if sample_type.lower() == "blood"

            else "Saliva"

        )


        # ----------------------------------------------------
        # CHECK REQUIRED 1000 GENES
        #
        # Gene_Symbol IS NOT required here.
        # The genes are expected to be individual columns.
        # ----------------------------------------------------

        (
            missing_genes,
            gene_column_map
        ) = check_patient_genes(
            df
        )


        if missing_genes:

         raise HTTPException(
         status_code=400,
         detail={
            "message": (
                "The patient CSV is missing "
                "genes required by the trained GNN."
            ),

            "missing_count": len(missing_genes),

            "required_genes": len(selected_genes),

            "available_gene_columns": len(gene_column_map),

            "total_csv_columns": len(df.columns),

            "example_missing_genes": missing_genes[:50],

            "first_available_columns": [
                str(column)
                for column in df.columns[:30]
            ],

            "solution": (
                "The patient CSV must contain expression "
                "values for the same 1000 genes used during "
                "GNN training."
            )
        }
    )


        # ----------------------------------------------------
        # EXTRACT GENE VALUES
        # ----------------------------------------------------

        gene_values = []


        for gene in selected_genes:

            actual_column = (
                gene_column_map[gene]
            )


            value = pd.to_numeric(
                patient[actual_column],
                errors="coerce"
            )


            if pd.isna(value):

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid or missing gene value "
                        f"for {gene}."
                    )
                )


            if not np.isfinite(
                float(value)
            ):

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid numeric value "
                        f"for gene {gene}."
                    )
                )


            gene_values.append(
                float(value)
            )


        # ----------------------------------------------------
        # BIOMARKERS
        # ----------------------------------------------------

        biomarkers = {}


        # Create case-insensitive biomarker lookup
        biomarker_lookup = {

            str(column)
            .strip()
            .upper():

            column

            for column
            in df.columns

        }


        for column in BIOMARKER_COLUMNS:

            column_key = column.upper()


            if column_key in biomarker_lookup:

                actual_column = (
                    biomarker_lookup[
                        column_key
                    ]
                )


                value = pd.to_numeric(
                    patient[actual_column],
                    errors="coerce"
                )


                if pd.isna(value):

                    biomarkers[column] = None

                else:

                    biomarkers[column] = round(
                        float(value),
                        4
                    )

            else:

                biomarkers[column] = None


        # ----------------------------------------------------
        # RUN GNN
        # ----------------------------------------------------

        prediction_result = (
            run_gnn_prediction(
                gene_values
            )
        )


        # ----------------------------------------------------
        # TOP GENE SIGNALS
        # ----------------------------------------------------

        important_features = (
            get_top_gene_signals(
                gene_values
            )
        )


        # ----------------------------------------------------
        # OPTIONAL DIAGNOSIS
        #
        # ONLY returned for testing.
        # NEVER used for prediction.
        # ----------------------------------------------------

        provided_diagnosis = None


        diagnosis_lookup = {

            str(column)
            .strip()
            .upper():

            column

            for column
            in df.columns

        }


        if "DIAGNOSIS" in diagnosis_lookup:

            diagnosis_column = (
                diagnosis_lookup[
                    "DIAGNOSIS"
                ]
            )


            diagnosis_value = patient[
                diagnosis_column
            ]


            if not pd.isna(
                diagnosis_value
            ):

                provided_diagnosis = str(
                    diagnosis_value
                ).strip()


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {

            "status":
                "success",

            "dataset_type":
                "patient",

            "patient": {

                "patient_id":
                    patient_id,

                "sample_type":
                    sample_type

            },

            "biomarkers":
                biomarkers,

            "prediction": {

                "class":
                    prediction_result[
                        "prediction"
                    ],

                "class_name":
                    CLASS_DESCRIPTIONS[
                        prediction_result[
                            "prediction"
                        ]
                    ],

                "confidence":
                    prediction_result[
                        "confidence"
                    ],

                "probabilities":
                    prediction_result[
                        "probabilities"
                    ]

            },

            "important_features":
                important_features,

            "model": {

                "name":
                    "Weighted Improved GCN",

                "selected_genes":
                    len(selected_genes),

                "graph_type":
                    "k-NN",

                "k":
                    10,

                "graph_edges":
                    int(edge_index.shape[1])

            },

            "data_validation": {

                "required_gene_count":
                    len(selected_genes),

                "provided_gene_count":
                    len(gene_column_map),

                "missing_gene_count":
                    len(missing_genes),

                "gene_data_complete":
                    len(missing_genes) == 0

            },

            "provided_diagnosis":
                provided_diagnosis,

            "clinical_note":
                (
                    "This is a research prototype prediction "
                    "and must not be interpreted as a clinical "
                    "diagnosis."
                )

        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )