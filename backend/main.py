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
import gc

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
torch.set_num_threads(1)

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
# RESEARCH DATASET PREDICTION - MEMORY OPTIMIZED
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

        # ----------------------------------------------------
        # READ CSV DIRECTLY FROM UPLOAD
        # Avoid await file.read() because it creates another
        # large copy of the entire CSV in RAM.
        # ----------------------------------------------------

        uploaded_file = file.file

        # Read only the header first
        header_df = pd.read_csv(
            uploaded_file,
            nrows=0
        )

        header_df = clean_columns(
            header_df
        )

        if "Gene_Symbol" not in header_df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Research dataset must contain "
                    "'Gene_Symbol' column."
                )
            )

        sample_columns = [
            column
            for column in header_df.columns
            if column != "Gene_Symbol"
        ]

        if len(sample_columns) == 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "CSV must contain at least "
                    "one sample column."
                )
            )

        # ----------------------------------------------------
        # READ DATA WITH FLOAT32
        #
        # float64 = 8 bytes/value
        # float32 = 4 bytes/value
        #
        # This roughly halves expression-data memory.
        # ----------------------------------------------------

        uploaded_file.seek(0)

        dtype_map = {
            column: np.float32
            for column in sample_columns
        }

        df = pd.read_csv(
            uploaded_file,
            dtype=dtype_map
        )

        df = clean_columns(
            df
        )

        # ----------------------------------------------------
        # REMOVE DUPLICATE GENES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CHECK REQUIRED GENES
        # ----------------------------------------------------

        missing_genes = [

            gene

            for gene in selected_genes

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

        # ----------------------------------------------------
        # SELECT ONLY THE 1000 GENES NEEDED BY THE GNN
        # ----------------------------------------------------

        expression = df.loc[
            selected_genes
        ]

        # ----------------------------------------------------
        # CHECK FOR MISSING / NON-NUMERIC VALUES
        #
        # Because expression columns were read as float32,
        # invalid values become NaN.
        # ----------------------------------------------------

        if expression.isna().any().any():

            bad_columns = expression.columns[
                expression.isna().any()
            ]

            bad_sample = str(
                bad_columns[0]
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    "Non-numeric or missing values "
                    f"found in sample {bad_sample}."
                )
            )

        # ----------------------------------------------------
        # RUN GNN
        # ----------------------------------------------------

        results = []

        for sample_id in expression.columns:

            sample_values = (
                expression[sample_id]
                .to_numpy(dtype=np.float32)
            )

            prediction_result = (
                run_gnn_prediction(
                    sample_values
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
        # DATASET BASIC INFORMATION
        # ----------------------------------------------------

        number_of_genes = len(
            df.index
        )

        number_of_samples = len(
            df.columns
        )

        # ----------------------------------------------------
        # DATASET STATISTICS
        #
        # Work directly on the original float32 dataframe.
        # Do NOT create another numeric_df copy.
        # ----------------------------------------------------

        missing_values = int(
            df.isna()
            .sum()
            .sum()
        )

        mean_expression = float(
            df.to_numpy(dtype=np.float32)
            .mean()
        )

        # ----------------------------------------------------
        # TOP VARIABLE GENES
        # ----------------------------------------------------

        gene_variances = df.var(
            axis=1,
            ddof=1
        )

        top_variable_genes_series = (
            gene_variances
            .nlargest(10)
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
        #
        # Only take the first sample and first 30 genes.
        # ----------------------------------------------------

        expression_overview = []

        if number_of_samples > 0:

            first_sample = df.iloc[:, 0]

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

        # ----------------------------------------------------
        # FREE LARGE OBJECTS BEFORE RETURNING
        # ----------------------------------------------------

        del df
        del expression
        del gene_variances

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

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