# DataCouncil

**DataCouncil** is a multi-agent AI analytics platform that transforms raw CSV datasets into actionable business insights through collaborative AI agents.

Built for **Qwen Cloud AI Hackathon 2026 – Track 3: Agent Society**, DataCouncil demonstrates how specialized AI agents can collaborate to produce more reliable analytical results than a single general-purpose AI agent.

---

##  Overview

Instead of assigning every task to one AI model, DataCouncil decomposes the workflow into multiple specialized agents.

Each agent performs a dedicated responsibility and passes its output to the next agent, creating an explainable and collaborative AI pipeline.

---

##  System Architecture

![Architecture](images/architecture.png)

---

##  Multi-Agent Workflow

###  Data Quality Agent
- Detects missing values
- Identifies outliers
- Reviews dataset quality
- Reports structural issues

↓

### Data Analyst Agent
- Generates evidence-backed business insights
- Explains why each insight matters
- Uses the quality report as context

↓

###  Insight Validator Agent
- Independently reviews every analyst insight
- Labels each insight as:
  - ✅ CONFIRMED
  - ⚠️ FLAGGED
- Produces corrected versions for weak claims

↓

###  Visualization Agent
- Selects the most suitable visualization
- Recommends chart type
- Chooses appropriate dataset columns

↓

###  Report Writer Agent
- Produces an executive summary
- Uses only validated insights
- Summarizes findings for decision-makers

---

##  Single-Agent Baseline

To demonstrate the value of collaborative AI, DataCouncil also runs the same dataset through a **single general-purpose agent**.

The application compares both approaches using measurable metrics, including:

- Number of validated insights
- Weak insights detected
- Automatically revised claims
- Cross-agent disagreements
- Specialized agent collaboration

This satisfies the **Track 3: Agent Society** requirement of demonstrating measurable benefits over a single-agent approach.

---

##  Features

- Multi-agent collaboration using CrewAI
- Sequential task decomposition
- Cross-agent validation
- Business insight generation
- Automatic visualization recommendation
- Executive report generation
- Interactive Streamlit dashboard
- Plotly visualizations
- Single-agent baseline comparison
- Debug mode for rapid UI development

---

##  Technologies Used

- Python
- CrewAI
- Qwen API
- Streamlit
- Plotly
- Pandas

---

##  Project Structure

```text
DataCouncil/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── images/
│   └── architecture.png
│
├── outputs/
│   └── datacouncil_outputs.json
│
└── sample_data/
    └── sample_dataset.csv
```

---

##  Installation

Clone the repository

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

##  API Key

The application requires a valid **Qwen API Key**.

Enter your API key in the Streamlit sidebar before running the analysis.

**Do not hardcode API keys in the source code.**

---

##  Sample Dataset

A sample dataset is included  for demonstration purposes.

You may also upload your own CSV datasets directly through the application.

---

##  Application Workflow

1. Upload a CSV dataset
2. Click **Run DataCouncil Analysis**
3. The five AI agents collaborate sequentially
4. Review:
   - Data Quality Report
   - Business Insights
   - Validation Results
   - Recommended Visualization
   - Executive Summary
   - Baseline Comparison

---

##  Track 3 Requirements Addressed

✅ Multi-agent collaboration

✅ Specialized agent roles

✅ Task decomposition

✅ Agent-to-agent communication

✅ Validation and disagreement resolution

✅ Comparison against a single-agent baseline

---

##  License

This project is licensed under the MIT License.

---

##  Developed For

**Qwen Cloud AI Hackathon 2026**

Track 3 — Agent Society
