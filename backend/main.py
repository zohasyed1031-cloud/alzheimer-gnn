# ============================================================
# MEMORY OPTIMIZATION - MUST BE BEFORE NUMPY / TORCH
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ============================================================
# IMPORTS
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import io
import gc
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
    version="2.3.0"
)




app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://alzheimer-gnn.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
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

# Reduce CPU memory/thread usage on Render
torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch may already have initialized inter-op threads
    pass


# ============================================================
# LOAD SELECTED GENES
# ============================================================

selected_genes_df = pd.read_csv(
    MODEL_DIR / "selected_genes_1000.csv",
    usecols=["Gene_Symbol"]
)

selected_genes_df["Gene_Symbol"] = (
    selected_genes_df["Gene_Symbol"]
    .astype(str)
    .str.strip()
)

selected_genes = list(
    dict.fromkeys(
        selected_genes_df["Gene_Symbol"].tolist()
    )
)

del selected_genes_df

gc.collect()


if len(selected_genes) == 0:
    raise RuntimeError(
        "No selected genes were found in selected_genes_1000.csv."
    )


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    MODEL_DIR / "scaler.pkl"
)


# ============================================================
# LOAD GRAPH
# ============================================================

edge_index = torch.load(
    MODEL_DIR / "edge_index.pt",
    map_location="cpu",
    weights_only=True
)

edge_weight = torch.load(
    MODEL_DIR / "edge_weight.pt",
    map_location="cpu",
    weights_only=True
)

# Keep graph tensors on CPU for the CPU deployment
edge_index = edge_index.to(
    device=device,
    dtype=torch.long
)

edge_weight = edge_weight.to(
    device=device,
    dtype=torch.float32
)


# ============================================================
# MODEL
# ============================================================

class AlzheimerGCNWeighted(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(
            1,
            64
        )

        self.conv2 = GCNConv(
            64,
            64
        )

        self.dropout = nn.Dropout(
            0.3
        )

        self.fc = nn.Linear(
            128,
            3
        )


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

        x = F.relu(
            x
        )

        # Dropout is inactive because model.eval()
        x = self.dropout(
            x
        )

        x = self.conv2(
            x,
            edge_index,
            edge_weight=edge_weight
        )

        x = F.relu(
            x
        )

        mean_pool = global_mean_pool(
            x,
            batch
        )

        max_pool = global_max_pool(
            x,
            batch
        )

        x = torch.cat(
            [
                mean_pool,
                max_pool
            ],
            dim=1
        )

        x = self.fc(
            x
        )

        return x


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = AlzheimerGCNWeighted().to(
    device
)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "weighted_improved_gcn.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# Make sure parameters do not require gradients
for parameter in model.parameters():
    parameter.requires_grad_(False)


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

        "status":
            "running",

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

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# HELPER:
# CREATE CASE-INSENSITIVE COLUMN MAP
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

    column_lookup = create_gene_column_map(
        df
    )

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
            ] = column_lookup[
                gene_key
            ]

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
# FIND COLUMN CASE-INSENSITIVELY
# ============================================================

def find_column(
    df,
    possible_names
):

    column_lookup = create_gene_column_map(
        df
    )

    for name in possible_names:

        key = (
            str(name)
            .strip()
            .upper()
        )

        if key in column_lookup:

            return column_lookup[
                key
            ]

    return None


# ============================================================
# HELPER:
# RUN GNN FOR ONE SAMPLE
#
# MEMORY OPTIMIZED
# ============================================================

def run_gnn_prediction(
    sample_values
):

    if len(sample_values) != len(
        selected_genes
    ):

        raise ValueError(
            f"Expected {len(selected_genes)} gene values, "
            f"but received {len(sample_values)}."
        )


    # --------------------------------------------------------
    # Convert directly to NumPy float32
    # No Pandas DataFrame required
    # --------------------------------------------------------

    sample_array = np.asarray(
        sample_values,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Apply the same scaler used during training
    # --------------------------------------------------------

    sample_scaled = scaler.transform(
        sample_array
    )


    # Ensure float32 without unnecessary copy
    sample_scaled = np.asarray(
        sample_scaled,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Convert to graph node features
    # --------------------------------------------------------

    x = torch.from_numpy(
        sample_scaled[0]
    ).view(
        -1,
        1
    ).to(
        device
    )


    # --------------------------------------------------------
    # One graph = one batch
    # --------------------------------------------------------

    batch = torch.zeros(
        x.size(0),
        dtype=torch.long,
        device=device
    )


    # --------------------------------------------------------
    # INFERENCE
    #
    # inference_mode uses less overhead than no_grad
    # --------------------------------------------------------

    with torch.inference_mode():

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

        predicted_class = int(
            torch.argmax(
                probabilities
            ).item()
        )

        probability_values = {

            "AD":
                round(
                    float(
                        probabilities[0]
                    ) * 100,
                    2
                ),

            "CTL":
                round(
                    float(
                        probabilities[1]
                    ) * 100,
                    2
                ),

            "MCI":
                round(
                    float(
                        probabilities[2]
                    ) * 100,
                    2
                )

        }


    # --------------------------------------------------------
    # Convert result before deleting tensors
    # --------------------------------------------------------

    prediction = CLASS_NAMES[
        predicted_class
    ]

    confidence = probability_values[
        prediction
    ]


    # --------------------------------------------------------
    # Release temporary objects
    # --------------------------------------------------------

    del x
    del batch
    del output
    del probabilities
    del sample_array
    del sample_scaled

    gc.collect()


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
#
# MEMORY OPTIMIZED
# ============================================================

def get_top_gene_signals(
    sample_values
):

    sample_array = np.asarray(
        sample_values,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )


    # Apply training scaler
    scaled_values = scaler.transform(
        sample_array
    )[0]


    scaled_values = np.asarray(
        scaled_values,
        dtype=np.float32
    )


    signals = np.abs(
        scaled_values
    )


    # --------------------------------------------------------
    # Get indices of top 10 signals
    # --------------------------------------------------------

    top_indices = np.argsort(
        signals
    )[-10:][::-1]


    results = []


    for index in top_indices:

        results.append({

            "gene":
                str(
                    selected_genes[index]
                ),

            "signal":
                round(
                    float(
                        signals[index]
                    ),
                    4
                ),

            "expression":
                round(
                    float(
                        sample_values[index]
                    ),
                    4
                )

        })


    # --------------------------------------------------------
    # Free temporary arrays
    # --------------------------------------------------------

    del sample_array
    del scaled_values
    del signals
    del top_indices

    gc.collect()


    return results


# ============================================================
# PATIENT CSV PREDICTION
#
# Expected:
#
# Patient_ID | Sample_Type | 1000 genes | biomarkers...
#
# Biomarkers are optional.
# ============================================================

@app.post("/patient-predict")
async def patient_predict(
    file: UploadFile = File(...)
):

    try:

        # ====================================================
        # READ PATIENT CSV
        # ====================================================

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded patient CSV "
                    "is empty."
                )
            )


        df = pd.read_csv(
            io.BytesIO(
                contents
            )
        )


        del contents

        df = clean_columns(
            df
        )


        if df.empty:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Patient CSV does not contain "
                    "any data rows."
                )
            )


        # ====================================================
        # USE FIRST ROW
        # ====================================================

        patient_row = df.iloc[0]


        # ====================================================
        # CHECK REQUIRED GENES
        # ====================================================

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
                        "Patient CSV is missing "
                        "required gene columns."
                    ),

                    "missing_count":
                        len(
                            missing_genes
                        ),

                    "required_genes":
                        len(
                            selected_genes
                        ),

                    "example_missing_genes":
                        missing_genes[:30]

                }
            )


        # ====================================================
        # EXTRACT GENE VALUES
        # ====================================================

        sample_values = []

        invalid_genes = []


        for gene in selected_genes:

            column_name = gene_column_map[
                gene
            ]

            value = pd.to_numeric(
                patient_row[
                    column_name
                ],
                errors="coerce"
            )


            if pd.isna(
                value
            ):

                invalid_genes.append(
                    gene
                )

            else:

                sample_values.append(
                    float(
                        value
                    )
                )


        if invalid_genes:

            raise HTTPException(
                status_code=400,
                detail={

                    "message": (
                        "Patient CSV contains "
                        "missing or non-numeric "
                        "gene values."
                    ),

                    "invalid_count":
                        len(
                            invalid_genes
                        ),

                    "example_invalid_genes":
                        invalid_genes[:30]

                }
            )


        # ====================================================
        # PATIENT ID
        # ====================================================

        patient_id_column = find_column(
            df,
            [
                "Patient_ID",
                "Patient ID",
                "PatientID",
                "ID"
            ]
        )


        if patient_id_column is not None:

            patient_id_value = (
                patient_row[
                    patient_id_column
                ]
            )

            if pd.isna(
                patient_id_value
            ):

                patient_id = "Unknown"

            else:

                patient_id = str(
                    patient_id_value
                )

        else:

            patient_id = "Unknown"


        # ====================================================
        # SAMPLE TYPE
        # ====================================================

        sample_type_column = find_column(
            df,
            [
                "Sample_Type",
                "Sample Type",
                "SampleType",
                "Tissue",
                "Tissue_Type"
            ]
        )


        if sample_type_column is not None:

            sample_type_value = (
                patient_row[
                    sample_type_column
                ]
            )

            if pd.isna(
                sample_type_value
            ):

                sample_type = "Unknown"

            else:

                sample_type = str(
                    sample_type_value
                )

        else:

            sample_type = "Unknown"


        # ====================================================
        # BIOMARKER ANALYSIS
        # ====================================================

        column_lookup = (
            create_gene_column_map(
                df
            )
        )


        biomarkers = {}


        biomarker_aliases = {

            "AB42": [

                "AB42",
                "Aβ42",
                "Aβ-42",
                "Amyloid_Beta_42",
                "Amyloid Beta 42",
                "AmyloidBeta42"

            ],

            "AB40": [

                "AB40",
                "Aβ40",
                "Aβ-40",
                "Amyloid_Beta_40",
                "Amyloid Beta 40",
                "AmyloidBeta40"

            ],

            "pTau181": [

                "pTau181",
                "p-tau181",
                "p_tau181",
                "PTAU181",
                "pTau-181"

            ],

            "pTau217": [

                "pTau217",
                "p-tau217",
                "p_tau217",
                "PTAU217",
                "pTau-217"

            ],

            "AB42_AB40_Ratio": [

                "AB42_AB40_Ratio",
                "AB42/AB40",
                "Aβ42/Aβ40",
                "Amyloid_Beta_Ratio",
                "Amyloid Beta Ratio"

            ]

        }


        for biomarker in BIOMARKER_COLUMNS:

            matching_column = None


            for alias in biomarker_aliases[
                biomarker
            ]:

                alias_key = (
                    str(alias)
                    .strip()
                    .upper()
                )


                if alias_key in column_lookup:

                    matching_column = (
                        column_lookup[
                            alias_key
                        ]
                    )

                    break


            if matching_column is None:

                biomarkers[
                    biomarker
                ] = None

            else:

                value = pd.to_numeric(
                    patient_row[
                        matching_column
                    ],
                    errors="coerce"
                )


                if pd.isna(
                    value
                ):

                    biomarkers[
                        biomarker
                    ] = None

                else:

                    biomarkers[
                        biomarker
                    ] = round(
                        float(
                            value
                        ),
                        4
                    )


        # ====================================================
        # CALCULATE RATIO IF NOT PROVIDED
        # ====================================================

        if (

            biomarkers[
                "AB42_AB40_Ratio"
            ] is None

            and biomarkers[
                "AB42"
            ] is not None

            and biomarkers[
                "AB40"
            ] is not None

            and biomarkers[
                "AB40"
            ] != 0

        ):

            biomarkers[
                "AB42_AB40_Ratio"
            ] = round(

                biomarkers[
                    "AB42"
                ]
                /
                biomarkers[
                    "AB40"
                ],

                4

            )


        # ====================================================
        # RUN GNN
        # ====================================================

        prediction_result = (
            run_gnn_prediction(
                sample_values
            )
        )


        # ====================================================
        # IMPORTANT GENE SIGNALS
        # ====================================================

        important_features = (
            get_top_gene_signals(
                sample_values
            )
        )


        # ====================================================
        # OPTIONAL PROVIDED DIAGNOSIS
        # ====================================================

        diagnosis_column = find_column(
            df,
            [
                "Diagnosis",
                "Label",
                "Class",
                "Diagnosis_Label"
            ]
        )


        provided_diagnosis = None


        if diagnosis_column is not None:

            diagnosis_value = (
                patient_row[
                    diagnosis_column
                ]
            )


            if not pd.isna(
                diagnosis_value
            ):

                provided_diagnosis = str(
                    diagnosis_value
                )


        # ====================================================
        # DATA VALIDATION
        # ====================================================

        validation = {

            "required_gene_count":
                len(
                    selected_genes
                ),

            "provided_gene_count":
                len(
                    selected_genes
                ),

            "missing_gene_count":
                0,

            "invalid_gene_count":
                len(
                    invalid_genes
                ),

            "biomarkers_available":
                sum(
                    value is not None
                    for value
                    in biomarkers.values()
                )

        }


        # ====================================================
        # CLINICAL NOTE
        # ====================================================

        clinical_note = (
            "This result is generated by a research "
            "prototype using gene-expression data and "
            "should not be interpreted as a clinical diagnosis."
        )


        # ====================================================
        # PREPARE RESPONSE
        # ====================================================

        response = {

            "status":
                "success",

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

                "description":
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
                    len(
                        selected_genes
                    ),

                "graph_edges":
                    int(
                        edge_index.shape[1]
                    ),

                "device":
                    str(
                        device
                    )

            },

            "data_validation":
                validation,

            "provided_diagnosis":
                provided_diagnosis,

            "clinical_note":
                clinical_note

        }


        # ====================================================
        # CLEAN PATIENT OBJECTS
        # ====================================================

        del patient_row
        del gene_column_map
        del column_lookup
        del df
        del sample_values
        del invalid_genes

        gc.collect()


        return response


    except HTTPException:

        raise


    except Exception as e:

        gc.collect()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# RESEARCH DATASET PREDICTION
#
# Expected:
#
# Gene_Symbol | GSM001 | GSM002 | ...
#
# ============================================================

# ============================================================
# RESEARCH DATASET PREDICTION - CHUNKED FOR LOWER MEMORY
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        uploaded_file = file.file

        # Read the CSV header only
        header_df = pd.read_csv(uploaded_file, nrows=0)
        header_df = clean_columns(header_df)

        if "Gene_Symbol" not in header_df.columns:
            raise HTTPException(
                status_code=400,
                detail="Research dataset must contain 'Gene_Symbol' column."
            )

        sample_columns = [
            col for col in header_df.columns
            if col != "Gene_Symbol"
        ]

        if not sample_columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain at least one sample column."
            )

        number_of_samples = len(sample_columns)
        print(f"Research upload: {number_of_samples} samples detected", flush=True)
        del header_df
        gc.collect()

        uploaded_file.seek(0)

        # Store selected genes only; process the rest in chunks
        selected_gene_set = set(selected_genes)
        selected_expression = {}
        seen_genes = set()

        missing_values = 0
        expression_sum = 0.0
        valid_value_count = 0
        number_of_genes = 0

        top_variances = []
        expression_overview = []

        dtype_map = {
            col: np.float32 for col in sample_columns
        }

        chunks = pd.read_csv(
            uploaded_file,
            dtype=dtype_map,
            chunksize=500
        )

        del dtype_map
        gc.collect()

        for chunk in chunks:
            chunk = clean_columns(chunk)

            chunk["Gene_Symbol"] = (
                chunk["Gene_Symbol"]
                .astype(str)
                .str.strip()
            )

            # Preserve the first occurrence of each gene,
            # including duplicates that span different chunks.
            chunk = chunk.drop_duplicates(
                subset=["Gene_Symbol"]
            )

            new_gene_mask = ~chunk["Gene_Symbol"].isin(
                seen_genes
            )
            chunk = chunk.loc[new_gene_mask]

            if chunk.empty:
                del chunk
                continue

            gene_names = chunk["Gene_Symbol"].tolist()
            seen_genes.update(gene_names)
            number_of_genes += len(gene_names)

            values = chunk[sample_columns]

            # Accumulate dataset statistics
            missing_values += int(
                values.isna().sum().sum()
            )

            valid_count = int(values.count().sum())

            if valid_count:
                expression_sum += float(
                    values.sum().sum()
                )
                valid_value_count += valid_count

            # Keep the first 30 genes for the expression overview
            if len(expression_overview) < 30 and number_of_samples:
                first_sample = sample_columns[0]

                for gene, value in zip(
                    gene_names,
                    chunk[first_sample].tolist()
                ):
                    if len(expression_overview) >= 30:
                        break

                    if pd.notna(value):
                        expression_overview.append({
                            "gene": str(gene),
                            "expression": round(float(value), 4)
                        })

            # Track the ten genes with highest variance
            variances = values.var(axis=1, ddof=1)

            for gene, variance in zip(
                gene_names, variances.tolist()
            ):
                if pd.notna(variance):
                    top_variances.append(
                        (str(gene), float(variance))
                    )

            top_variances = sorted(
                top_variances,
                key=lambda item: item[1],
                reverse=True
            )[:10]

            # Retain expression values only for model genes
            selected_mask = chunk["Gene_Symbol"].isin(
                selected_gene_set
            )

            selected_rows = chunk.loc[selected_mask]

            for _, row in selected_rows.iterrows():
                gene = row["Gene_Symbol"]

                if gene not in selected_expression:
                    selected_expression[gene] = (
                        row[sample_columns]
                        .to_numpy(dtype=np.float32, copy=True)
                    )

            del values, variances, selected_rows, chunk
            gc.collect()

        # Check that all required model genes were found
        missing_genes = [
            gene for gene in selected_genes
            if gene not in selected_expression
        ]

        if missing_genes:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Uploaded dataset is missing required genes."
                    ),
                    "missing_count": len(missing_genes),
                    "required_genes": len(selected_genes),
                    "example_missing_genes": missing_genes[:30]
                }
            )

        # Reconstruct only the small 1000-gene expression matrix
        expression = np.stack(
            [selected_expression[gene] for gene in selected_genes],
            axis=0
        )

        del selected_expression
        gc.collect()

        # Validate values for genes used by the GNN
        if not np.isfinite(expression).all():
            del expression
            gc.collect()

            raise HTTPException(
                status_code=400,
                detail=(
                    "Required genes contain missing, non-numeric, "
                    "or non-finite values."
                )
            )

        # Predict one sample at a time
        print("CSV processing complete. Starting GNN predictions.", flush=True)
        results = []

        for sample_index, sample_id in enumerate(sample_columns):
            print(f"Predicting sample {sample_index + 1}/{number_of_samples}", flush=True)
            sample_values = expression[:, sample_index]

            prediction_result = run_gnn_prediction(
                sample_values
            )

            results.append({
                "sample_id": str(sample_id),
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "probabilities": prediction_result["probabilities"]
            })

        del expression
        gc.collect()

        mean_expression = (
            expression_sum / valid_value_count
            if valid_value_count
            else 0.0
        )

        top_variable_genes = [
            {
                "gene": gene,
                "variance": round(variance, 6)
            }
            for gene, variance in top_variances
        ]

        return {
            "status": "success",
            "dataset_type": "research",
            "sample_count": len(results),
            "number_of_genes": number_of_genes,
            "number_of_samples": number_of_samples,
            "missing_values": missing_values,
            "mean_expression": round(mean_expression, 4),
            "top_variable_genes": top_variable_genes,
            "expression_overview": expression_overview,
            "model": "Weighted Improved GCN",
            "selected_gene_count": len(selected_genes),
            "graph_edges": int(edge_index.shape[1]),
            "k_value": 10,
            "results": results
        }

    except HTTPException:
        raise

    except Exception as e:
        gc.collect()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
