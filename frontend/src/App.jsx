import React, { useState } from "react";

import {
  Brain,
  Droplets,
  Activity,
  Upload,
  FileText,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Users,
  Dna,
  BarChart3,
  Database,
  Network,
  ChevronRight,
  FlaskConical,
  BrainCircuit,
  Target,
  Gauge,
  RefreshCw,
  UserRound,
  Microscope,
  CircleDot,
  FileSpreadsheet
} from "lucide-react";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell
} from "recharts";


const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";


function App() {

  // ========================================================
  // STATE
  // ========================================================

  const [activePage, setActivePage] = useState("dashboard");

  const [sampleType, setSampleType] = useState("blood");

  const [file, setFile] = useState(null);

  const [patientReport, setPatientReport] = useState(null);

  const [loading, setLoading] = useState(false);

  const [predictionLoading, setPredictionLoading] = useState(false);

  const [result, setResult] = useState(null);

  const [patientResult, setPatientResult] = useState(null);

  const [error, setError] = useState("");

  const [predictionError, setPredictionError] = useState("");


  // ========================================================
  // PAGE NAVIGATION
  // ========================================================

  const navigate = (page) => {

    setActivePage(page);

    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  };


  // ========================================================
  // RESEARCH DATASET FILE
  // ========================================================

  const handleFileChange = (event) => {

    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    setFile(selectedFile);

    setError("");

    setResult(null);
  };


  // ========================================================
  // PATIENT REPORT FILE
  // ========================================================

  const handlePatientReportChange = (event) => {

    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    setPatientReport(selectedFile);

    setPredictionError("");

    setPatientResult(null);
  };


  // ========================================================
  // RESEARCH DATASET ANALYSIS
  // ========================================================

  const handleAnalyze = async () => {

    if (!file) {

      setError(
        "Please upload a research dataset first."
      );

      return;
    }

    setLoading(true);

    setError("");

    try {

      const formData = new FormData();

      formData.append(
        "file",
        file
      );

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData
        }
      );

      const data = await response.json();

      if (!response.ok) {

        throw new Error(
          data?.detail?.message ||
          data?.detail ||
          "Dataset analysis failed."
        );
      }

      setResult(data);

      navigate("analysis");

    }

    catch (err) {

      setError(
        err.message ||
        "Unable to analyze dataset."
      );

    }

    finally {

      setLoading(false);

    }
  };


  // ========================================================
  // PATIENT PREDICTION
  // ========================================================

  const handlePatientAnalysis = async () => {

    if (!patientReport) {

      setPredictionError(
        "Please upload the patient's CSV report."
      );

      return;
    }

    setPredictionLoading(true);

    setPredictionError("");

    try {

      const formData = new FormData();

      formData.append(
        "file",
        patientReport
      );

      const response = await fetch(
        `${API_URL}/patient-predict`,
        {
          method: "POST",
          body: formData
        }
      );

      const data = await response.json();

      if (!response.ok) {

        throw new Error(
          data?.detail?.message ||
          data?.detail ||
          "Patient analysis failed."
        );
      }

      setPatientResult(data);

      navigate("patient");

    }

    catch (err) {

      setPredictionError(
        err.message ||
        "Unable to analyze patient report."
      );

    }

    finally {

      setPredictionLoading(false);

    }
  };


  // ========================================================
  // RESEARCH GNN
  // ========================================================

  const handleRunGNN = async () => {

    if (!file) {

      setPredictionError(
        "Please upload a research dataset first."
      );

      return;
    }

    setPredictionLoading(true);

    setPredictionError("");

    try {

      const formData = new FormData();

      formData.append(
        "file",
        file
      );

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData
        }
      );

      const data = await response.json();

      if (!response.ok) {

        throw new Error(
          data?.detail?.message ||
          data?.detail ||
          "GNN prediction failed."
        );
      }

      setResult(data);

      navigate("gnn");

    }

    catch (err) {

      setPredictionError(
        err.message ||
        "Unable to run GNN."
      );

    }

    finally {

      setPredictionLoading(false);

    }
  };


  // ========================================================
  // DERIVED DATA
  // ========================================================

  const variableGenes =
    result?.top_variable_genes || [];

  const expressionOverview =
    result?.expression_overview || [];

  const predictionResults =
    result?.results || [];

  const geneCount =
    result?.number_of_genes ?? 0;

  const sampleCount =
    result?.number_of_samples ?? 0;

  const missingValues =
    result?.missing_values ?? 0;

  const meanExpression =
    result?.mean_expression ?? 0;

  const graphEdgeCount =
    result?.graph_edges ?? 18282;

  const selectedGeneCount =
    result?.selected_gene_count ?? 1000;

  const kValue =
    result?.k_value ?? 10;

  const firstPrediction =
    predictionResults.length > 0
      ? predictionResults[0]
      : null;


  // ========================================================
  // HELPERS
  // ========================================================

  const predictionClass = (prediction) => {

    if (prediction === "AD")
      return "danger";

    if (prediction === "MCI")
      return "warning";

    return "success";
  };


  // ========================================================
  // PATIENT PROBABILITY DATA
  // ========================================================

  const probabilityData =
    patientResult?.prediction?.probabilities
      ? [
          {
            name: "AD",
            value:
              patientResult.prediction
                .probabilities.AD
          },
          {
            name: "MCI",
            value:
              patientResult.prediction
                .probabilities.MCI
          },
          {
            name: "CTL",
            value:
              patientResult.prediction
                .probabilities.CTL
          }
        ]
      : [];


  // ========================================================
  // STYLES
  // ========================================================

  const styles = `

  * {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    font-family:
      Inter,
      -apple-system,
      BlinkMacSystemFont,
      "Segoe UI",
      sans-serif;

    background: #f5f7fb;
    color: #172033;
  }

  button,
  input {
    font: inherit;
  }

  .app {
    min-height: 100vh;
  }

  /* ------------------------------------------------------
     SIDEBAR
  ------------------------------------------------------ */

  .sidebar {
    position: fixed;
    left: 0;
    top: 0;
    bottom: 0;

    width: 250px;

    background:
      linear-gradient(
        180deg,
        #101a35 0%,
        #162348 100%
      );

    color: white;

    padding: 25px 18px;

    z-index: 10;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;

    padding: 0 10px 28px;

    border-bottom:
      1px solid
      rgba(255,255,255,.1);
  }

  .brand-icon {
    width: 42px;
    height: 42px;

    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #5968e8;
  }

  .brand-title {
    font-weight: 750;
    font-size: 16px;
  }

  .brand-subtitle {
    font-size: 11px;
    opacity: .55;
    margin-top: 3px;
  }

  .nav {
    margin-top: 25px;
  }

  .nav-button {
    width: 100%;

    display: flex;
    align-items: center;
    gap: 12px;

    border: 0;
    background: transparent;

    color: rgba(255,255,255,.65);

    padding: 13px 14px;
    margin-bottom: 6px;

    border-radius: 10px;

    cursor: pointer;

    text-align: left;

    transition: .2s;
  }

  .nav-button:hover {
    background:
      rgba(255,255,255,.07);

    color: white;
  }

  .nav-button.active {
    background: #5968e8;
    color: white;
  }

  .sidebar-note {
    position: absolute;
    left: 18px;
    right: 18px;
    bottom: 22px;

    background:
      rgba(255,255,255,.06);

    border:
      1px solid
      rgba(255,255,255,.08);

    border-radius: 12px;

    padding: 13px;

    font-size: 11px;

    color:
      rgba(255,255,255,.55);
  }

  /* ------------------------------------------------------
     MAIN
  ------------------------------------------------------ */

  .main {
    margin-left: 250px;
    min-height: 100vh;
  }

  .topbar {
    height: 70px;

    background: white;

    border-bottom:
      1px solid #e9ecf3;

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 0 35px;
  }

  .page-title {
    font-size: 15px;
    font-weight: 700;
  }

  .top-status {
    display: flex;
    align-items: center;
    gap: 8px;

    font-size: 12px;
    color: #5f6b82;
  }

  .status-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #35b879;
  }

  .content {
    padding: 35px;
    max-width: 1500px;
  }

  /* ------------------------------------------------------
     HERO
  ------------------------------------------------------ */

  .hero {
    background:
      linear-gradient(
        120deg,
        #18254d,
        #354da2
      );

    border-radius: 20px;

    padding: 40px;

    color: white;

    display: flex;
    justify-content: space-between;
    align-items: center;

    overflow: hidden;

    position: relative;
  }

  .hero-content {
    max-width: 700px;
    position: relative;
    z-index: 2;
  }

  .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;

    padding: 6px 10px;

    border-radius: 20px;

    background:
      rgba(255,255,255,.1);

    font-size: 11px;
    font-weight: 650;

    margin-bottom: 17px;
  }

  .hero h1 {
    margin: 0;

    font-size: 35px;
    line-height: 1.15;
    letter-spacing: -.8px;
  }

  .hero p {
    color:
      rgba(255,255,255,.75);

    max-width: 620px;

    line-height: 1.7;

    font-size: 14px;

    margin: 17px 0 25px;
  }

  .hero-actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }

  .hero-decoration {
    width: 230px;
    height: 230px;

    border-radius: 50%;

    border:
      30px solid
      rgba(255,255,255,.05);

    display: flex;
    align-items: center;
    justify-content: center;

    margin-right: 50px;
  }

  .hero-decoration-inner {
    width: 110px;
    height: 110px;

    border-radius: 50%;

    background:
      rgba(255,255,255,.08);

    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* ------------------------------------------------------
     BUTTONS
  ------------------------------------------------------ */

  .button {
    border: 0;

    border-radius: 9px;

    padding: 11px 17px;

    display: inline-flex;
    align-items: center;
    gap: 8px;

    font-size: 13px;
    font-weight: 650;

    cursor: pointer;

    transition: .2s;
  }

  .button-primary {
    background: white;
    color: #25376f;
  }

  .button-primary:hover {
    transform: translateY(-1px);
  }

  .button-dark {
    background: #26386f;
    color: white;
  }

  .button-blue {
    background: #5968e8;
    color: white;
  }

  .button-light {
    background: #eef1f8;
    color: #35415b;
  }

  .button:disabled {
    opacity: .5;
    cursor: not-allowed;
  }

  /* ------------------------------------------------------
     SECTION
  ------------------------------------------------------ */

  .section {
    margin-top: 28px;
  }

  .section-header {
    margin-bottom: 17px;
  }

  .section-title {
    font-size: 18px;
    font-weight: 750;
    margin: 0;
  }

  .section-subtitle {
    font-size: 12px;
    color: #778197;
    margin-top: 5px;
  }

  /* ------------------------------------------------------
     CARDS
  ------------------------------------------------------ */

  .cards {
    display: grid;
    grid-template-columns:
      repeat(3, 1fr);

    gap: 17px;
  }

  .cards-four {
    display: grid;
    grid-template-columns:
      repeat(4, 1fr);

    gap: 15px;
  }

  .card {
    background: white;

    border:
      1px solid #e7eaf1;

    border-radius: 14px;

    padding: 21px;

    box-shadow:
      0 3px 12px
      rgba(20,30,60,.035);
  }

  .card-icon {
    width: 42px;
    height: 42px;

    border-radius: 11px;

    background: #eef1ff;

    color: #5968e8;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-bottom: 15px;
  }

  .card h3 {
    margin: 0 0 8px;

    font-size: 14px;
  }

  .card p {
    margin: 0;

    color: #788398;

    font-size: 12px;

    line-height: 1.6;
  }

  .stat-value {
    font-size: 25px;
    font-weight: 800;
  }

  .stat-label {
    color: #788398;
    font-size: 11px;
    margin-top: 5px;
  }

  /* ------------------------------------------------------
     UPLOAD
  ------------------------------------------------------ */

  .upload-grid {
    display: grid;
    grid-template-columns:
      1.3fr .7fr;

    gap: 20px;
  }

  .upload-box {
    width: 100%;
    min-height: 205px;

    background: #fafbfe;

    border:
      1.5px dashed #cdd3e1;

    border-radius: 15px;

    padding: 32px 25px;

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;

    text-align: center;

    cursor: pointer;

    transition:
      border-color .2s,
      background .2s,
      transform .2s;

    margin-top: 4px;
  }

  .upload-box:hover {
    border-color: #5968e8;
    background: #f7f8ff;
    transform: translateY(-1px);
  }

  .upload-icon {
    width: 54px;
    height: 54px;

    border-radius: 14px;

    margin: 0 0 13px;

    background: #eef1ff;
    color: #5968e8;

    display: flex;
    align-items: center;
    justify-content: center;
  }

  .upload-title {
    font-size: 14px;
    font-weight: 700;
  }

  .upload-description {
    color: #80899c;

    font-size: 11px;

    margin-top: 7px;

    line-height: 1.5;

    max-width: 420px;
  }

  .file-name {
    margin-top: 15px;

    display: inline-flex;

    align-items: center;

    justify-content: center;

    gap: 7px;

    padding: 8px 12px;

    background: #eef8f2;

    color: #29845c;

    border-radius: 8px;

    font-size: 11px;

    max-width: 90%;

    overflow: hidden;

    text-overflow: ellipsis;

    white-space: nowrap;
  }

  input[type="file"] {
    display: none;
  }

  /* ------------------------------------------------------
     SAMPLE TYPE
  ------------------------------------------------------ */

  .sample-selector {
    width: 100%;

    display: flex;

    gap: 12px;

    margin-bottom: 20px;
  }

  .sample-option {
    flex: 1;

    border:
      1px solid #e2e6ef;

    background: white;

    border-radius: 12px;

    padding: 15px;

    cursor: pointer;

    display: flex;
    align-items: center;
    gap: 10px;

    transition: .2s;
  }

  .sample-option.active {
    border-color: #5968e8;

    background: #f4f5ff;

    color: #4352ca;
  }

  .sample-option-title {
    font-weight: 700;
    font-size: 13px;
  }

  .sample-option-subtitle {
    color: #8790a3;
    font-size: 10px;
    margin-top: 3px;
  }

  /* ------------------------------------------------------
     CHARTS
  ------------------------------------------------------ */

  .chart-grid {
    display: grid;

    grid-template-columns:
      repeat(2, 1fr);

    gap: 18px;
  }

  .chart-card {
    background: white;

    border:
      1px solid #e7eaf1;

    border-radius: 14px;

    padding: 20px;
  }

  .chart-header {
    display: flex;
    justify-content: space-between;

    margin-bottom: 18px;
  }

  .chart-title {
    font-size: 13px;
    font-weight: 750;
  }

  .chart-subtitle {
    font-size: 10px;
    color: #858ea0;
    margin-top: 4px;
  }

  .chart {
    width: 100%;
    height: 300px;
  }

  /* ------------------------------------------------------
     PIPELINE
  ------------------------------------------------------ */

  .pipeline {
    display: grid;

    grid-template-columns:
      repeat(4, 1fr);

    gap: 12px;
  }

  .pipeline-step {
    position: relative;

    background: white;

    border:
      1px solid #e7eaf1;

    border-radius: 12px;

    padding: 18px;
  }

  .pipeline-number {
    width: 27px;
    height: 27px;

    border-radius: 8px;

    background: #eef1ff;

    color: #5968e8;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 11px;
    font-weight: 800;

    margin-bottom: 12px;
  }

  .pipeline-step h4 {
    margin: 0 0 6px;

    font-size: 12px;
  }

  .pipeline-step p {
    margin: 0;

    font-size: 10px;

    line-height: 1.5;

    color: #7b8599;
  }

  /* ------------------------------------------------------
     RESULT
  ------------------------------------------------------ */

  .result-banner {
    border-radius: 16px;

    padding: 25px;

    background: white;

    border:
      1px solid #e7eaf1;

    display: flex;

    align-items: center;

    justify-content: space-between;
  }

  .result-label {
    color: #7c869b;
    font-size: 11px;
    margin-bottom: 5px;
  }

  .result-class {
    font-size: 30px;
    font-weight: 850;
  }

  .result-class.success {
    color: #258b61;
  }

  .result-class.warning {
    color: #bd801b;
  }

  .result-class.danger {
    color: #c34d59;
  }

  .confidence {
    text-align: right;
  }

  .confidence-value {
    font-size: 28px;
    font-weight: 800;
  }

  .confidence-label {
    font-size: 10px;
    color: #7c869b;
  }

  .probability-list {
    margin-top: 20px;
  }

  .probability-row {
    margin-bottom: 15px;
  }

  .probability-top {
    display: flex;
    justify-content: space-between;

    font-size: 12px;

    margin-bottom: 6px;
  }

  .probability-track {
    height: 8px;

    background: #edf0f5;

    border-radius: 10px;

    overflow: hidden;
  }

  .probability-fill {
    height: 100%;

    border-radius: 10%;

    background: #5968e8;
  }

  .probability-fill.ad {
    background: #d86670;
  }

  .probability-fill.mci {
    background: #d9a443;
  }

  .probability-fill.ctl {
    background: #3aa878;
  }

  /* ------------------------------------------------------
     BIOMARKERS
  ------------------------------------------------------ */

  .biomarker-grid {
    display: grid;

    grid-template-columns:
      repeat(5, 1fr);

    gap: 12px;
  }

  .biomarker {
    background: white;

    border:
      1px solid #e7eaf1;

    border-radius: 12px;

    padding: 17px;
  }

  .biomarker-name {
    font-size: 10px;
    color: #818b9f;
  }

  .biomarker-value {
    margin-top: 7px;

    font-size: 18px;

    font-weight: 800;
  }

  /* ------------------------------------------------------
     PATIENT HEADER
  ------------------------------------------------------ */

  .patient-header {
    background:
      linear-gradient(
        120deg,
        #ffffff,
        #f4f6ff
      );

    border:
      1px solid #e3e7f2;

    border-radius: 16px;

    padding: 22px;

    display: flex;

    justify-content: space-between;

    align-items: center;
  }

  .patient-info {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .patient-avatar {
    width: 50px;
    height: 50px;

    border-radius: 13px;

    background: #5968e8;

    color: white;

    display: flex;
    align-items: center;
    justify-content: center;
  }

  .patient-id {
    font-weight: 800;
    font-size: 16px;
  }

  .patient-type {
    color: #7e879b;
    font-size: 11px;
    margin-top: 4px;
  }

  .badge {
    display: inline-flex;
    align-items: center;

    padding: 7px 11px;

    border-radius: 20px;

    font-size: 10px;
    font-weight: 700;
  }

  .badge-blood {
    background: #fff0f1;
    color: #b74b55;
  }

  .badge-saliva {
    background: #eef8f7;
    color: #28827b;
  }

  /* ------------------------------------------------------
     TABLE
  ------------------------------------------------------ */

  .table-container {
    overflow-x: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th {
    text-align: left;

    font-size: 10px;

    color: #788297;

    padding: 11px;

    border-bottom:
      1px solid #e6e9f0;
  }

  td {
    padding: 12px 11px;

    font-size: 11px;

    border-bottom:
      1px solid #f0f2f6;
  }

  .table-prediction {
    font-weight: 750;
  }

  /* ------------------------------------------------------
     INFO
  ------------------------------------------------------ */

  .info-box {
    padding: 14px 16px;

    border-radius: 11px;

    background: #f3f6ff;

    border:
      1px solid #dce3ff;

    color: #5d6781;

    font-size: 11px;

    line-height: 1.6;
  }

  .warning-box {
    padding: 14px 16px;

    border-radius: 11px;

    background: #fff8e9;

    border:
      1px solid #f1dfaf;

    color: #806a36;

    font-size: 11px;

    line-height: 1.6;
  }

  .error-box {
    padding: 14px 16px;

    border-radius: 11px;

    background: #fff0f1;

    border:
      1px solid #f1c9ce;

    color: #a9434c;

    font-size: 11px;

    line-height: 1.6;

    margin-top: 15px;
  }

  /* ------------------------------------------------------
     FOOTER
  ------------------------------------------------------ */

  .footer {
    text-align: center;

    color: #929bad;

    font-size: 10px;

    padding: 35px 0 10px;
  }

  /* ------------------------------------------------------
     RESPONSIVE
  ------------------------------------------------------ */

  @media(max-width: 1050px) {

    .sidebar {
      width: 210px;
    }

    .main {
      margin-left: 210px;
    }

    .cards-four {
      grid-template-columns:
        repeat(2, 1fr);
    }

    .biomarker-grid {
      grid-template-columns:
        repeat(3, 1fr);
    }

  }


  @media(max-width: 800px) {

    .sidebar {
      position: relative;

      width: 100%;

      height: auto;
    }

    .sidebar-note {
      display: none;
    }

    .main {
      margin-left: 0;
    }

    .nav {
      display: flex;
      overflow-x: auto;
    }

    .nav-button {
      min-width: 130px;
    }

    .hero-decoration {
      display: none;
    }

    .content {
      padding: 20px;
    }

    .cards,
    .chart-grid,
    .upload-grid,
    .pipeline {
      grid-template-columns: 1fr;
    }

    .biomarker-grid {
      grid-template-columns:
        repeat(2, 1fr);
    }

  }

  `;


  // ========================================================
  // NAVIGATION
  // ========================================================

  const navItems = [

    {
      id: "dashboard",
      label: "Dashboard",
      icon: <Brain size={17} />
    },

    {
      id: "datasets",
      label: "Research Datasets",
      icon: <Database size={17} />
    },

    {
      id: "analysis",
      label: "Dataset Analysis",
      icon: <BarChart3 size={17} />
    },

    {
      id: "gnn",
      label: "GNN Model",
      icon: <Network size={17} />
    }

  ];


  // ========================================================
  // SIDEBAR
  // ========================================================

  const Sidebar = () => (

    <aside className="sidebar">

      <div className="brand">

        <div className="brand-icon">
          <BrainCircuit size={23} />
        </div>

        <div>

          <div className="brand-title">
            NeuroGraph AI
          </div>

          <div className="brand-subtitle">
            Alzheimer's Research Platform
          </div>

        </div>

      </div>


      <nav className="nav">

        {navItems.map((item) => (

          <button
            key={item.id}
            className={
              `nav-button ${
                activePage === item.id
                  ? "active"
                  : ""
              }`
            }
            onClick={() =>
              navigate(item.id)
            }
          >

            {item.icon}

            <span>
              {item.label}
            </span>

          </button>

        ))}

      </nav>


      <div className="sidebar-note">

        <ShieldCheck
          size={13}
          style={{
            verticalAlign: "middle",
            marginRight: 5
          }}
        />

        Research prototype only.
        Results require clinical validation.

      </div>

    </aside>

  );


  // ========================================================
  // TOP BAR
  // ========================================================

  const Topbar = () => {

    let title = "Patient Analysis";

    if (activePage === "datasets")
      title = "Research Datasets";

    if (activePage === "analysis")
      title = "Dataset Analysis";

    if (activePage === "gnn")
      title = "GNN Model";

    if (activePage === "patient")
      title = "Patient Analysis Report";


    return (

      <header className="topbar">

        <div className="page-title">
          {title}
        </div>

        <div className="top-status">

          <span className="status-dot" />

          GNN API Connected

        </div>

      </header>

    );

  };


  // ========================================================
  // DASHBOARD
  // ========================================================

  const DashboardPage = () => (

    <>

      <section className="hero">

        <div className="hero-content">

          <div className="eyebrow">

            <Microscope size={13} />

            AI-assisted research analysis

          </div>


          <h1>
            Early Alzheimer's Detection
            Through Gene Expression
          </h1>


          <p>
            Upload a patient's blood or saliva
            gene-expression report and analyze
            the molecular profile using the
            trained graph neural network.
          </p>


          <div className="hero-actions">

            <button
              className="button button-primary"
              onClick={() =>
                document
                  .getElementById(
                    "patient-upload"
                  )
                  ?.scrollIntoView({
                    behavior: "smooth"
                  })
              }
            >

              <UserRound size={15} />

              Analyze Patient

              <ArrowRight size={15} />

            </button>


            <button
              className="button"
              style={{
                background:
                  "rgba(255,255,255,.12)",
                color: "white"
              }}
              onClick={() =>
                navigate("gnn")
              }
            >

              <Network size={15} />

              Explore GNN Model

            </button>

          </div>

        </div>


        <div className="hero-decoration">

          <div className="hero-decoration-inner">

            <Dna size={52} />

          </div>

        </div>

      </section>


      <section
        className="section"
        id="patient-upload"
      >

        <div className="section-header">

          <h2 className="section-title">
            Analyze Patient
          </h2>

          <div className="section-subtitle">
            Upload one patient CSV containing
            biomarkers and the selected gene-expression
            features.
          </div>

        </div>


        <div
          className="card"
          style={{
            padding: "24px"
          }}
        >

          <div className="sample-selector">

            <div
              className={
                `sample-option ${
                  sampleType === "blood"
                    ? "active"
                    : ""
                }`
              }
              onClick={() =>
                setSampleType("blood")
              }
            >

              <Droplets size={20} />

              <div>

                <div className="sample-option-title">
                  Blood Sample
                </div>

                <div className="sample-option-subtitle">
                  Blood gene-expression profile
                </div>

              </div>

            </div>


            <div
              className={
                `sample-option ${
                  sampleType === "saliva"
                    ? "active"
                    : ""
                }`
              }
              onClick={() =>
                setSampleType("saliva")
              }
            >

              <Activity size={20} />

              <div>

                <div className="sample-option-title">
                  Saliva Sample
                </div>

                <div className="sample-option-subtitle">
                  Saliva gene-expression profile
                </div>

              </div>

            </div>

          </div>


          <label
            htmlFor="patientFile"
            className="upload-box"
          >

            <div className="upload-icon">
              <Upload size={24} />
            </div>


            <div className="upload-title">
              Upload Patient CSV
            </div>


            <div className="upload-description">
              Patient_ID, Sample_Type, biomarkers
              and GNN gene-expression features
            </div>


            {patientReport && (

              <div className="file-name">

                <FileSpreadsheet size={13} />

                {patientReport.name}

                <CheckCircle2 size={13} />

              </div>

            )}

          </label>


          <input
            id="patientFile"
            type="file"
            accept=".csv"
            onChange={
              handlePatientReportChange
            }
          />


          {predictionError && (

            <div className="error-box">

              <AlertCircle
                size={13}
                style={{
                  verticalAlign: "middle",
                  marginRight: 5
                }}
              />

              {predictionError}

            </div>

          )}


          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              marginTop: 17
            }}
          >

            <button
              className="button button-blue"
              disabled={
                predictionLoading ||
                !patientReport
              }
              onClick={
                handlePatientAnalysis
              }
            >

              {predictionLoading ? (

                <>

                  <RefreshCw
                    size={14}
                    className="spin"
                  />

                  Analyzing...

                </>

              ) : (

                <>

                  <Gauge size={14} />

                  Analyze Patient

                  <ArrowRight size={14} />

                </>

              )}

            </button>

          </div>

        </div>

      </section>


      <section className="section">

        <div className="section-header">

          <h2 className="section-title">
            Analysis Pipeline
          </h2>

          <div className="section-subtitle">
            From patient sample to GNN prediction
          </div>

        </div>


        <div className="pipeline">

          {[
            [
              "1",
              "Patient CSV",
              "Blood or saliva report is uploaded."
            ],
            [
              "2",
              "Feature Analysis",
              "Biomarkers and gene-expression values are extracted."
            ],
            [
              "3",
              "Graph Neural Network",
              "1000 selected genes are represented as graph nodes."
            ],
            [
              "4",
              "Prediction",
              "AD, MCI and control probabilities are generated."
            ]
          ].map((step) => (

            <div
              className="pipeline-step"
              key={step[0]}
            >

              <div className="pipeline-number">
                {step[0]}
              </div>

              <h4>
                {step[1]}
              </h4>

              <p>
                {step[2]}
              </p>

            </div>

          ))}

        </div>

      </section>


      <section className="section">

        <div className="cards">

          <div className="card">

            <div className="card-icon">
              <Dna size={21} />
            </div>

            <h3>
              Gene Expression
            </h3>

            <p>
              The patient report contains the
              gene-expression values required by
              the trained model.
            </p>

          </div>


          <div className="card">

            <div className="card-icon">
              <Network size={21} />
            </div>

            <h3>
              Graph Neural Network
            </h3>

            <p>
              A weighted k-NN graph connects the
              selected genes before GCN processing.
            </p>

          </div>


          <div className="card">

            <div className="card-icon">
              <Target size={21} />
            </div>

            <h3>
              Model Output
            </h3>

            <p>
              The system presents AD, MCI and
              control model probabilities.
            </p>

          </div>

        </div>

      </section>

    </>

  );


  // ========================================================
  // PATIENT RESULT PAGE
  // ========================================================

  const PatientPage = () => {

    if (!patientResult) {

      return (

        <div className="card">

          <h3>
            No patient analysis available
          </h3>

          <p>
            Upload a patient CSV from the
            dashboard to generate a result.
          </p>

        </div>

      );

    }


    const prediction =
      patientResult.prediction;


    const predictionType =
      predictionClass(
        prediction.class
      );


    return (

      <>

        <div className="patient-header">

          <div className="patient-info">

            <div className="patient-avatar">
              <UserRound size={23} />
            </div>

            <div>

              <div className="patient-id">
                {patientResult.patient.patient_id}
              </div>

              <div className="patient-type">

                {patientResult.patient.sample_type}
                {" "}sample • GNN analysis

              </div>

            </div>

          </div>


          <div
            className={
              `badge ${
                patientResult.patient.sample_type
                  .toLowerCase() === "blood"
                  ? "badge-blood"
                  : "badge-saliva"
              }`
            }
          >

            {patientResult.patient.sample_type}

          </div>

        </div>


        <section className="section">

          <div className="result-banner">

            <div>

              <div className="result-label">
                Model Prediction
              </div>

              <div
                className={
                  `result-class ${predictionType}`
                }
              >
                {prediction.class}
              </div>

              <div
                style={{
                  fontSize: 11,
                  color: "#7c869b",
                  marginTop: 4
                }}
              >
                {prediction.class_name}
              </div>

            </div>


            <div className="confidence">

              <div className="confidence-value">
                {prediction.confidence}%
              </div>

              <div className="confidence-label">
                Model confidence
              </div>

            </div>

          </div>

        </section>


        <section className="section">

          <div className="section-header">

            <h2 className="section-title">
              Biomarker Analysis
            </h2>

            <div className="section-subtitle">
              Values extracted from the uploaded
              patient report
            </div>

          </div>


          <div className="biomarker-grid">

            {[
              ["Aβ42", "AB42"],
              ["Aβ40", "AB40"],
              ["p-Tau181", "pTau181"],
              ["p-Tau217", "pTau217"],
              ["Aβ42 / Aβ40", "AB42_AB40_Ratio"]
            ].map(([label, key]) => (

              <div
                className="biomarker"
                key={key}
              >

                <div className="biomarker-name">
                  {label}
                </div>

                <div className="biomarker-value">

                  {
                    patientResult.biomarkers[key]
                      ?? "—"
                  }

                </div>

              </div>

            ))}

          </div>

        </section>


        <section className="section">

          <div className="chart-grid">

            <div className="chart-card">

              <div className="chart-header">

                <div>

                  <div className="chart-title">
                    Prediction Probability
                  </div>

                  <div className="chart-subtitle">
                    Model output across three classes
                  </div>

                </div>

              </div>


              <div className="chart">

                <ResponsiveContainer
                  width="100%"
                  height="100%"
                >

                  <PieChart>

                    <Pie
                      data={probabilityData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      innerRadius={58}
                      label
                    >

                      {probabilityData.map(
                        (entry, index) => {

                          const colors = {
                            AD: "#d86670",
                            MCI: "#d9a443",
                            CTL: "#3aa878"
                          };

                          return (
                            <Cell
                              key={index}
                              fill={colors[entry.name]}
                            />
                          );

                        }
                      )}

                    </Pie>

                    <Tooltip />

                  </PieChart>

                </ResponsiveContainer>

              </div>

            </div>


            <div className="chart-card">

              <div className="chart-header">

                <div>

                  <div className="chart-title">
                    Class Probabilities
                  </div>

                  <div className="chart-subtitle">
                    Weighted GCN prediction
                  </div>

                </div>

              </div>


              <div className="probability-list">

                {[
                  ["AD", prediction.probabilities.AD],
                  ["MCI", prediction.probabilities.MCI],
                  ["CTL", prediction.probabilities.CTL]
                ].map(([label, value]) => (

                  <div
                    className="probability-row"
                    key={label}
                  >

                    <div className="probability-top">

                      <span>
                        {label === "CTL"
                          ? "Control"
                          : label}
                      </span>

                      <strong>
                        {value}%
                      </strong>

                    </div>


                    <div className="probability-track">

                      <div
                        className={
                          `probability-fill ${label.toLowerCase()}`
                        }
                        style={{
                          width: `${value}%`
                        }}
                      />

                    </div>

                  </div>

                ))}

              </div>

            </div>

          </div>

        </section>


        <section className="section">

          <div className="chart-card">

            <div className="chart-header">

              <div>

                <div className="chart-title">
                  Important Gene Signals
                </div>

                <div className="chart-subtitle">
                  Highest standardized signal values
                  among the selected genes
                </div>

              </div>

            </div>


            <div className="chart">

              <ResponsiveContainer
                width="100%"
                height="100%"
              >

                <BarChart
                  data={
                    patientResult.important_features
                  }
                  margin={{
                    top: 5,
                    right: 15,
                    left: 0,
                    bottom: 45
                  }}
                >

                  <CartesianGrid
                    strokeDasharray="3 3"
                  />

                  <XAxis
                    dataKey="gene"
                    angle={-45}
                    textAnchor="end"
                    interval={0}
                  />

                  <YAxis />

                  <Tooltip />

                  <Bar
                    dataKey="signal"
                    fill="#5968e8"
                    radius={[5,5,0,0]}
                  />

                </BarChart>

              </ResponsiveContainer>

            </div>

          </div>

        </section>


        <section className="section">

          <div className="info-box">

            <ShieldCheck
              size={14}
              style={{
                verticalAlign: "middle",
                marginRight: 6
              }}
            />

            <strong>
              Research prototype:
            </strong>{" "}

            The displayed probability is the
            output of the trained GNN model. It
            should not be interpreted as a clinical
            diagnosis or validated medical risk
            score.

          </div>

        </section>

      </>

    );

  };


  // ========================================================
  // DATASETS PAGE
  // ========================================================

  const DatasetsPage = () => (

    <>

      <div className="section-header">

        <h2 className="section-title">
          Research Dataset Analysis
        </h2>

        <div className="section-subtitle">
          Upload a gene-expression research dataset
          for preprocessing analysis and GNN evaluation.
        </div>

      </div>


      <div className="card">

        <div className="info-box">

          <Database
            size={14}
            style={{
              verticalAlign: "middle",
              marginRight: 6
            }}
          />

          Expected research format:

          <strong>
            Gene_Symbol
          </strong>

          followed by sample columns such as
          GSM1539080, GSM1539081, etc.

        </div>


        <label
          htmlFor="researchFile"
          className="upload-box"
          style={{
            marginTop: 20
          }}
        >

          <div className="upload-icon">
            <Database size={24} />
          </div>

          <div className="upload-title">
            Upload Research Dataset
          </div>

          <div className="upload-description">
            CSV containing Gene_Symbol and sample
            expression columns
          </div>


          {file && (

            <div className="file-name">

              <FileSpreadsheet size={13} />

              {file.name}

              <CheckCircle2 size={13} />

            </div>

          )}

        </label>


        <input
          id="researchFile"
          type="file"
          accept=".csv"
          onChange={
            handleFileChange
          }
        />


        {error && (

          <div className="error-box">

            <AlertCircle
              size={13}
              style={{
                verticalAlign: "middle",
                marginRight: 5
              }}
            />

            {error}

          </div>

        )}


        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            marginTop: 17
          }}
        >

          <button
            className="button button-blue"
            disabled={
              loading ||
              !file
            }
            onClick={
              handleAnalyze
            }
          >

            {loading ? (

              <>

                <RefreshCw size={14} />

                Analyzing...

              </>

            ) : (

              <>

                <BarChart3 size={14} />

                Analyze Dataset

                <ArrowRight size={14} />

              </>

            )}

          </button>

        </div>

      </div>


      <section className="section">

        <div className="pipeline">

          {[
            [
              "1",
              "Gene Validation",
              "Checks whether the required genes are available."
            ],
            [
              "2",
              "Feature Analysis",
              "Calculates expression statistics and variable genes."
            ],
            [
              "3",
              "Graph Construction",
              "Uses the saved 1000-node k-NN graph."
            ],
            [
              "4",
              "GNN Evaluation",
              "Generates AD, MCI and CTL model predictions."
            ]
          ].map((step) => (

            <div
              className="pipeline-step"
              key={step[0]}
            >

              <div className="pipeline-number">
                {step[0]}
              </div>

              <h4>
                {step[1]}
              </h4>

              <p>
                {step[2]}
              </p>

            </div>

          ))}

        </div>

      </section>

    </>

  );


  // ========================================================
  // ANALYSIS PAGE
  // ========================================================

  const AnalysisPage = () => (

    <>

      {!result ? (

        <div className="card">

          <div className="card-icon">
            <BarChart3 size={21} />
          </div>

          <h3>
            No dataset analysis available
          </h3>

          <p>
            Upload a research dataset from the
            Datasets page to view its analysis.
          </p>

          <button
            className="button button-blue"
            style={{
              marginTop: 15
            }}
            onClick={() =>
              navigate("datasets")
            }
          >

            Upload Dataset

            <ArrowRight size={14} />

          </button>

        </div>

      ) : (

        <>

          <div className="cards-four">

            <div className="card">

              <div className="stat-value">
                {geneCount.toLocaleString()}
              </div>

              <div className="stat-label">
                Genes
              </div>

            </div>


            <div className="card">

              <div className="stat-value">
                {sampleCount.toLocaleString()}
              </div>

              <div className="stat-label">
                Samples
              </div>

            </div>


            <div className="card">

              <div className="stat-value">
                {missingValues.toLocaleString()}
              </div>

              <div className="stat-label">
                Missing Values
              </div>

            </div>


            <div className="card">

              <div className="stat-value">
                {meanExpression}
              </div>

              <div className="stat-label">
                Mean Expression
              </div>

            </div>

          </div>


          <section className="section">

            <div className="chart-grid">

              <div className="chart-card">

                <div className="chart-header">

                  <div>

                    <div className="chart-title">
                      Top Variable Genes
                    </div>

                    <div className="chart-subtitle">
                      Genes with highest expression
                      variability
                    </div>

                  </div>

                  <Activity size={18} />

                </div>


                <div className="chart">

                  <ResponsiveContainer
                    width="100%"
                    height="100%"
                  >

                    <BarChart
                      data={
                        variableGenes
                      }
                      margin={{
                        top: 5,
                        right: 15,
                        left: 0,
                        bottom: 55
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="gene"
                        angle={-45}
                        textAnchor="end"
                        interval={0}
                      />

                      <YAxis />

                      <Tooltip />

                      <Bar
                        dataKey="variance"
                        fill="#5968e8"
                        radius={[5,5,0,0]}
                      />

                    </BarChart>

                  </ResponsiveContainer>

                </div>

              </div>


              <div className="chart-card">

                <div className="chart-header">

                  <div>

                    <div className="chart-title">
                      Expression Overview
                    </div>

                    <div className="chart-subtitle">
                      First sample — first 30 genes
                    </div>

                  </div>

                  <Dna size={18} />

                </div>


                <div className="chart">

                  <ResponsiveContainer
                    width="100%"
                    height="100%"
                  >

                    <AreaChart
                      data={
                        expressionOverview
                      }
                      margin={{
                        top: 5,
                        right: 15,
                        left: 0,
                        bottom: 55
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="gene"
                        angle={-45}
                        textAnchor="end"
                        interval={0}
                      />

                      <YAxis />

                      <Tooltip />

                      <Area
                        type="monotone"
                        dataKey="expression"
                        fill="#5968e8"
                        stroke="#5968e8"
                        fillOpacity={0.18}
                      />

                    </AreaChart>

                  </ResponsiveContainer>

                </div>

              </div>

            </div>

          </section>


          <section className="section">

            <div className="card">

              <h3>
                Quality Checks
              </h3>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    "repeat(3,1fr)",
                  gap: 12,
                  marginTop: 15
                }}
              >

                <div className="info-box">

                  <CheckCircle2
                    size={13}
                    style={{
                      verticalAlign: "middle",
                      marginRight: 5
                    }}
                  />

                  Dataset loaded

                </div>


                <div className="info-box">

                  <CheckCircle2
                    size={13}
                    style={{
                      verticalAlign: "middle",
                      marginRight: 5
                    }}
                  />

                  Gene validation passed

                </div>


                <div className="info-box">

                  <CheckCircle2
                    size={13}
                    style={{
                      verticalAlign: "middle",
                      marginRight: 5
                    }}
                  />

                  Graph ready

                </div>

              </div>

            </div>

          </section>


          <section className="section">

            <div className="card">

              <h3>
                Next Step
              </h3>

              <p>
                The validated dataset can now be
                passed through the saved weighted
                GCN model.
              </p>

              <button
                className="button button-blue"
                style={{
                  marginTop: 15
                }}
                onClick={
                  handleRunGNN
                }
              >

                <BrainCircuit size={14} />

                Run GNN Model

                <ArrowRight size={14} />

              </button>

            </div>

          </section>

        </>

      )}

    </>

  );


  // ========================================================
  // GNN PAGE
  // ========================================================

  const GNNPage = () => (

    <>

      <div className="cards-four">

        <div className="card">

          <div className="stat-value">
            1,000
          </div>

          <div className="stat-label">
            Selected Genes
          </div>

        </div>


        <div className="card">

          <div className="stat-value">
            k-NN
          </div>

          <div className="stat-label">
            Graph Type
          </div>

        </div>


        <div className="card">

          <div className="stat-value">
            K = 10
          </div>

          <div className="stat-label">
            Neighbours
          </div>

        </div>


        <div className="card">

          <div className="stat-value">
            3
          </div>

          <div className="stat-label">
            Output Classes
          </div>

        </div>

      </div>


      <section className="section">

        <div className="card">

          <div className="section-header">

            <h2 className="section-title">
              Weighted Improved GCN
            </h2>

            <div className="section-subtitle">
              Gene-expression graph classification
              architecture
            </div>

          </div>


          <div className="pipeline">

            {[
              [
                "1",
                "Gene Features",
                "1000 selected genes with one expression value per node."
              ],
              [
                "2",
                "GCN Layer 1",
                "Transforms 1-dimensional node features into 64-dimensional representations."
              ],
              [
                "3",
                "GCN Layer 2",
                "Learns higher-level graph representations."
              ],
              [
                "4",
                "Pooling + FC",
                "Mean/max pooling followed by three-class classification."
              ]
            ].map((step) => (

              <div
                className="pipeline-step"
                key={step[0]}
              >

                <div className="pipeline-number">
                  {step[0]}
                </div>

                <h4>
                  {step[1]}
                </h4>

                <p>
                  {step[2]}
                </p>

              </div>

            ))}

          </div>

        </div>

      </section>


      <section className="section">

        <div className="cards">

          <div className="card">

            <div className="card-icon">
              <CircleDot size={21} />
            </div>

            <h3>
              Graph Nodes
            </h3>

            <p>
              Each of the 1,000 selected genes
              represents a node in the graph.
            </p>

          </div>


          <div className="card">

            <div className="card-icon">
              <Network size={21} />
            </div>

            <h3>
              k-NN Connections
            </h3>

            <p>
              Each gene is connected to its
              nearest neighbours using K = 10.
            </p>

          </div>


          <div className="card">

            <div className="card-icon">
              <BrainCircuit size={21} />
            </div>

            <h3>
              Classification
            </h3>

            <p>
              The graph representation is
              classified into AD, MCI or CTL.
            </p>

          </div>

        </div>

      </section>


      <section className="section">

        <div className="card">

          <div className="chart-header">

            <div>

              <div className="chart-title">
                Graph Information
              </div>

              <div className="chart-subtitle">
                Saved graph used by the trained model
              </div>

            </div>

          </div>


          <div className="cards-four">

            <div>

              <div className="stat-value">
                1,000
              </div>

              <div className="stat-label">
                Nodes
              </div>

            </div>


            <div>

              <div className="stat-value">
                {graphEdgeCount.toLocaleString()}
              </div>

              <div className="stat-label">
                Directed Edges
              </div>

            </div>


            <div>

              <div className="stat-value">
                18.282
              </div>

              <div className="stat-label">
                Average Degree
              </div>

            </div>


            <div>

              <div className="stat-value">
                0
              </div>

              <div className="stat-label">
                Zero-Degree Nodes
              </div>

            </div>

          </div>

        </div>

      </section>


      <section className="section">

        <div className="card">

          <h3>
            Research Dataset GNN Evaluation
          </h3>

          <p>
            Run the trained GNN on the currently
            uploaded research dataset.
          </p>


          {predictionError && (

            <div className="error-box">
              {predictionError}
            </div>

          )}


          <button
            className="button button-blue"
            style={{
              marginTop: 15
            }}
            disabled={
              predictionLoading ||
              !file
            }
            onClick={
              handleRunGNN
            }
          >

            <BrainCircuit size={14} />

            {predictionLoading
              ? "Running GNN..."
              : "Run GNN"}

          </button>

        </div>

      </section>


      {firstPrediction && (

        <section className="section">

          <div className="result-banner">

            <div>

              <div className="result-label">
                First Dataset Sample
              </div>

              <div
                className={
                  `result-class ${
                    predictionClass(
                      firstPrediction.prediction
                    )
                  }`
                }
              >
                {firstPrediction.prediction}
              </div>

              <div
                style={{
                  fontSize: 11,
                  color: "#7c869b"
                }}
              >
                {firstPrediction.sample_id}
              </div>

            </div>


            <div className="confidence">

              <div className="confidence-value">
                {firstPrediction.confidence}%
              </div>

              <div className="confidence-label">
                Model confidence
              </div>

            </div>

          </div>

        </section>

      )}


      {predictionResults.length > 0 && (

        <section className="section">

          <div className="card">

            <div className="section-header">

              <h2 className="section-title">
                Dataset Predictions
              </h2>

              <div className="section-subtitle">
                Model output for each uploaded sample
              </div>

            </div>


            <div className="table-container">

              <table>

                <thead>

                  <tr>

                    <th>
                      Sample
                    </th>

                    <th>
                      Prediction
                    </th>

                    <th>
                      Confidence
                    </th>

                    <th>
                      AD
                    </th>

                    <th>
                      MCI
                    </th>

                    <th>
                      CTL
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {predictionResults.map(
                    (item) => (

                      <tr
                        key={
                          item.sample_id
                        }
                      >

                        <td>
                          {item.sample_id}
                        </td>

                        <td
                          className={
                            `table-prediction ${
                              predictionClass(
                                item.prediction
                              )
                            }`
                          }
                        >
                          {item.prediction}
                        </td>

                        <td>
                          {item.confidence}%
                        </td>

                        <td>
                          {item.probabilities.AD}%
                        </td>

                        <td>
                          {item.probabilities.MCI}%
                        </td>

                        <td>
                          {item.probabilities.CTL}%
                        </td>

                      </tr>

                    )
                  )}

                </tbody>

              </table>

            </div>

          </div>

        </section>

      )}

    </>

  );


  // ========================================================
  // PAGE ROUTER
  // ========================================================

  const renderPage = () => {

    if (activePage === "dashboard")
      return <DashboardPage />;

    if (activePage === "datasets")
      return <DatasetsPage />;

    if (activePage === "analysis")
      return <AnalysisPage />;

    if (activePage === "gnn")
      return <GNNPage />;

    if (activePage === "patient")
      return <PatientPage />;

    return <DashboardPage />;
  };


  // ========================================================
  // APP
  // ========================================================

  return (

    <>

      <style>
        {styles}
      </style>


      <div className="app">

        <Sidebar />


        <main className="main">

          <Topbar />


          <div className="content">

            {renderPage()}


            <div className="footer">

              NeuroGraph AI • Alzheimer's
              Gene Expression Research Platform

              <br />

              Research prototype — not intended
              for clinical diagnosis.

            </div>

          </div>

        </main>

      </div>

    </>

  );

}


export default App;

