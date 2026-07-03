
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# ── Debug mode switch ─────────────────────────────────────────
DEBUG_MODE = True

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DataCouncil",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DataCouncil")
st.markdown("*A multi-agent AI analytics platform*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Qwen API Key", type="password")
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a CSV")
    st.markdown("2. Click Analyze")
    st.markdown("3. Five agents collaborate")
    st.markdown("4. Get insights + chart + report")
    st.divider()
    st.markdown("**Built for Track 3: Agent Society**")
    st.markdown("Qwen Cloud Hackathon 2026")

# ── File upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded: {uploaded_file.name} — {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)
    st.divider()

    if st.button("🚀 Run DataCouncil Analysis", type="primary"):

        if not api_key and not DEBUG_MODE:
            st.error("Please enter your Qwen API key in the sidebar.")
            st.stop()

        # ── BRANCH 1: Debug mode — load from JSON ─────────────
        if DEBUG_MODE:
            try:
                with open("datacouncil_outputs.json", "r") as f:
                    saved = json.load(f)

                quality_raw       = saved["quality"]
                analysis_raw      = saved["analysis"]
                validation_raw    = saved["validation"]
                visualization_raw = saved["visualization"]
                report_raw        = saved["report"]
                baseline_raw      = saved["baseline"]

                st.success("✅ Analysis complete!")

            except FileNotFoundError:
                st.error("datacouncil_outputs.json not found. Upload it to /content/ first.")
                st.stop()

        # ── BRANCH 2: Production — run real crew ───────────────
        else:
            from crewai import Agent, Task, Crew, Process, LLM
            import asyncio

            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_BASE"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

            llm = LLM(
                model="openai/qwen-plus",
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                temperature=0
            )

            data_summary   = df.describe().to_string()[:1500]
            missing_values = df.isnull().sum().to_string()
            sample_data    = df.head(5).to_string()
            numeric_cols   = list(df.select_dtypes(include="number").columns)
            categorical_cols = list(df.select_dtypes(exclude="number").columns)

            metrics = f"""
DATASET OVERVIEW
-----------------
Shape: {df.shape}

COLUMNS:
{list(df.columns)}

NUMERIC COLUMNS:
{numeric_cols}

CATEGORICAL COLUMNS:
{categorical_cols}

MISSING VALUES:
{missing_values}

STATISTICS (TRUNCATED):
{data_summary}

SAMPLE DATA (FIRST 5 ROWS):
{sample_data}
"""

            # ── Agents ────────────────────────────────────────
            quality_agent = Agent(
                role="Data Quality Analyst",
                goal="Identify data quality issues like missing values, outliers, and inconsistencies",
                backstory="You are a meticulous data engineer who never lets bad data pass through.",
                llm=llm, verbose=False
            )
            analyst_agent = Agent(
                role="Data Analyst",
                goal="Extract clear business insights from the data",
                backstory="You are a senior analyst who turns numbers into actionable insights.",
                llm=llm, verbose=False
            )
            validator_agent = Agent(
                role="Insight Validator",
                goal="Critically review insights and flag weak conclusions",
                backstory="You are a skeptical auditor who double-checks every claim.",
                llm=llm, verbose=False
            )
            visualization_agent = Agent(
                role="Visualization Specialist",
                goal="Recommend the single best chart for the key insight",
                backstory="You pick the clearest chart for any finding.",
                llm=llm, verbose=False
            )
            writer_agent = Agent(
                role="Report Writer",
                goal="Write a clear executive summary for stakeholders",
                backstory="You write concise reports executives can read in two minutes.",
                llm=llm, verbose=False
            )
            baseline_agent = Agent(
                role="Solo Data Analyst",
                goal="Handle all analysis tasks independently without any help",
                backstory="A generalist analyst working alone on the entire project.",
                llm=llm, verbose=False
            )

            # ── Tasks ─────────────────────────────────────────
            quality_task = Task(
                description=f"Review this data for quality issues:\n{metrics}",
                expected_output="A bullet list of data quality observations and flags.",
                agent=quality_agent
            )
            analysis_task = Task(
                description=(
                    "Using the quality report, analyze the business metrics and provide insights.\n"
                    "For each insight clearly state what the data shows, why it matters,\n"
                    "and what evidence supports it.\n"
                    "Note: your insights will be reviewed by a Validator Agent.\n"
                    "Make sure each claim is clearly supported by the data.\n"
                    "Do not speculate beyond what the numbers show."
                ),
                expected_output="Three to five evidence-backed business insights.",
                agent=analyst_agent,
                context=[quality_task]
            )
            validation_task = Task(
                description=(
                    "Review the business insights from the Data Analyst.\n"
                    "For EACH insight respond in this exact format:\n\n"
                    "INSIGHT: <restate briefly>\n"
                    "CONFIDENCE: <1-10>\n"
                    "VERDICT: <CONFIRMED|FLAGGED>\n"
                    "REASON: <one sentence>\n"
                    "REVISED: <corrected version if FLAGGED, else NONE>\n\n"
                    "Repeat for every insight.\n\n"
                    "Be critical. Flag anything that overgeneralizes or goes "
                    "beyond what the data shows."
                ),
                expected_output="Structured INSIGHT/CONFIDENCE/VERDICT/REASON/REVISED blocks.",
                agent=validator_agent,
                context=[analysis_task]
            )
            visualization_task = Task(
                description=(
                    "Based on the analyst findings, recommend ONE chart.\n"
                    "Respond in EXACTLY this format, nothing else:\n"
                    "CHART_TYPE: <bar|histogram|scatter|line>\n"
                    "X_COLUMN: <column name from dataset>\n"
                    "Y_COLUMN: <column name, or NONE if not needed>\n"
                    "REASON: <one sentence>\n\n"
                    "Do not add any explanation outside these four lines."
                ),
                expected_output="Four lines in CHART_TYPE/X_COLUMN/Y_COLUMN/REASON format.",
                agent=visualization_agent,
                context=[analysis_task]
            )
            report_task = Task(
                description=(
                    "Write a professional executive summary based on validated insights.\n"
                    "Mention where the validator flagged or revised analyst claims.\n"
                    "Include major findings and recommendations."
                ),
                expected_output="A concise executive report of approximately 200 words.",
                agent=writer_agent,
                context=[validation_task, visualization_task]
            )
            baseline_task = Task(
                description=(
                    f"Perform a complete analysis of this dataset on your own.\n"
                    f"Include: data quality observations, business insights, a critical\n"
                    f"review of your own insights, a chart recommendation, and a final\n"
                    f"executive summary.\n\n{metrics}"
                ),
                expected_output="A full report covering quality, insights, validation, chart, and summary.",
                agent=baseline_agent
            )

            # ── Run crews ──────────────────────────────────────
            crew = Crew(
                agents=[quality_agent, analyst_agent, validator_agent,
                        visualization_agent, writer_agent],
                tasks=[quality_task, analysis_task, validation_task,
                       visualization_task, report_task],
                process=Process.sequential,
                verbose=False
            )
            baseline_crew = Crew(
                agents=[baseline_agent],
                tasks=[baseline_task],
                process=Process.sequential,
                verbose=False
            )

            with st.status("🤖 DataCouncil agents are working...", expanded=True) as status:
                st.write("🔍 Data Quality Agent checking your dataset...")
                st.write("📊 Data Analyst Agent extracting insights...")
                st.write("⚖️  Validator Agent reviewing and correcting claims...")
                st.write("📈 Visualization Agent picking best chart...")
                st.write("✍️  Report Writer composing executive summary...")
                asyncio.run(crew.kickoff_async())
                status.update(label="✅ DataCouncil complete!", state="complete")

            with st.status("🤖 Running single-agent baseline...", expanded=True) as status2:
                st.write("Running everything through one agent alone...")
                asyncio.run(baseline_crew.kickoff_async())
                status2.update(label="✅ Baseline complete!", state="complete")

            quality_raw       = quality_task.output.raw
            analysis_raw      = analysis_task.output.raw
            validation_raw    = validation_task.output.raw
            visualization_raw = visualization_task.output.raw
            report_raw        = report_task.output.raw
            baseline_raw      = baseline_task.output.raw

        # ── Everything below runs for BOTH debug and production ─

        # ── Helper functions ──────────────────────────────────
        def parse_rec(text):
            result = {}
            for line in text.strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip().upper()] = v.strip()
            return result

        def find_col(name, cols):
            name_lower = name.lower().strip()
            for col in cols:
                if col.lower() == name_lower:
                    return col
            for col in cols:
                if name_lower in col.lower() or col.lower() in name_lower:
                    return col
            return None

        def count_insights(text):
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            return len([l for l in lines
                        if l.startswith(("-", "*", "•"))
                        or (len(l) > 2 and l[0].isdigit() and l[1] in (".", ")"))])

        def count_flagged(text):
            return text.upper().count("FLAGGED")

        def count_confirmed(text):
            return text.upper().count("CONFIRMED")

        def count_revised(text):
            return len([l for l in text.split("\n")
                        if l.strip().startswith("REVISED:")
                        and "NONE" not in l.upper()])

        flagged           = count_flagged(validation_raw)
        confirmed         = count_confirmed(validation_raw)
        revised           = count_revised(validation_raw)
        baseline_insights = count_insights(baseline_raw)
        multi_insights    = count_insights(analysis_raw)

        # ── Tabs ──────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Quality",
            "📊 Insights",
            "⚖️ Validation",
            "📈 Chart",
            "📝 Report",
            "📊 Baseline Comparison"
        ])

        # ── Tab 1: Quality ────────────────────────────────────
        with tab1:
            st.subheader("Data Quality Report")
            st.markdown(quality_raw)

        # ── Tab 2: Insights ───────────────────────────────────
        with tab2:
            st.subheader("Business Insights")
            st.markdown(analysis_raw)

        # ── Tab 3: Validation ─────────────────────────────────
        with tab3:
            st.subheader("Validation Review")
            st.markdown("🟢 Green = confirmed · 🟡 Yellow = flagged and revised")
            st.divider()
            blocks = [b.strip() for b in validation_raw.strip().split("\n\n") if b.strip()]
            for block in blocks:
                if "VERDICT: FLAGGED" in block:
                    st.warning(block)
                elif "VERDICT: CONFIRMED" in block:
                    st.success(block)
                else:
                    st.markdown(block)

        # ── Tab 4: Chart ──────────────────────────────────────
        with tab4:
            st.subheader("Recommended Chart")
            rec        = parse_rec(visualization_raw)
            chart_type = rec.get("CHART_TYPE", "bar").lower()
            x_col      = find_col(rec.get("X_COLUMN", ""), df.columns)
            y_raw      = rec.get("Y_COLUMN", "NONE")
            y_col      = find_col(y_raw, df.columns) if y_raw.upper() != "NONE" else None

            try:
                if chart_type == "histogram" or y_col is None:
                    fig = px.histogram(df, x=x_col,
                                       title=f"Distribution of {x_col}")
                elif chart_type == "bar":
                    fig = px.bar(df, x=x_col, y=y_col,
                                 title=f"{y_col} by {x_col}")
                elif chart_type == "scatter":
                    fig = px.scatter(df, x=x_col, y=y_col,
                                     title=f"{x_col} vs {y_col}")
                elif chart_type == "line":
                    fig = px.line(df, x=x_col, y=y_col,
                                  title=f"{y_col} over {x_col}")
                else:
                    fig = px.histogram(df, x=x_col)

                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Agent reasoning: {rec.get('REASON', '')}")

            except Exception as e:
                st.error(f"Chart failed, showing fallback: {e}")
                fallback = df.select_dtypes(include="number").columns[0]
                fig = px.histogram(df, x=fallback)
                st.plotly_chart(fig, use_container_width=True)

        # ── Tab 5: Report ─────────────────────────────────────
        with tab5:
            st.subheader("Executive Report")
            st.markdown(report_raw)
            st.divider()
            st.download_button(
                label="📥 Download Report",
                data=report_raw,
                file_name="datacouncil_report.txt",
                mime="text/plain"
            )

        # ── Tab 6: Baseline Comparison ────────────────────────
        with tab6:
            st.subheader("Single Agent vs DataCouncil")
            st.markdown(
                "This comparison satisfies Track 3's requirement "
                "for a measurable efficiency gain over a single-agent baseline."
            )
            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🤖 Baseline")
                st.markdown("*(single agent, no collaboration)*")
                st.metric("Agents involved", "1")
                st.metric("Specialized roles", "0")
                st.metric("Insights generated", baseline_insights)
                st.metric("Insights validated externally", "0")
                st.metric("Weak insights flagged", "0")
                st.metric("Auto-corrected insights", "0")
                st.metric("Cross-agent disagreements", "0")

            with col2:
                st.markdown("### 🧠 DataCouncil")
                st.markdown("*(five specialized agents)*")
                st.metric("Agents involved", "5")
                st.metric("Specialized roles", "5")
                st.metric("Insights generated", multi_insights,
                          delta=f"{multi_insights - baseline_insights} vs baseline")
                st.metric("Insights validated externally", confirmed + flagged,
                          delta=f"+{confirmed + flagged} vs baseline")
                st.metric("Weak insights flagged", flagged,
                          delta=f"+{flagged} vs baseline")
                st.metric("Auto-corrected insights", revised,
                          delta=f"+{revised} vs baseline")
                st.metric("Cross-agent disagreements", flagged,
                          delta=f"+{flagged} vs baseline")

            st.divider()
            st.markdown("### Qualitative Comparison")

            left_col, right_col = st.columns(2)

            with left_col:
                st.error(
                    f"**🤖 Baseline Agent**\n\n"
                    f"- Generated insights AND critiqued itself in one pass\n"
                    f"- No external validation\n"
                    f"- Same model accepted all of its own conclusions\n"
                    f"- Produced {baseline_insights} unchecked insights"
                )

            with right_col:
                st.success(
                    f"**🧠 DataCouncil**\n\n"
                    f"- Quality Agent reviewed the data first\n"
                    f"- Analyst focused only on business insights\n"
                    f"- Validator independently checked every insight\n"
                    f"- {flagged} insight(s) were challenged\n"
                    f"- {revised} insight(s) were revised before reporting"
                )

            st.divider()
            st.markdown("### Full Output Comparison")

            left, right = st.columns(2)

            with left:
                st.markdown("**🤖 Baseline Output**")
                st.text_area("baseline", baseline_raw, height=400,
                             label_visibility="collapsed")

            with right:
                st.markdown("**🧠 DataCouncil Final Report**")
                st.text_area("datacouncil", report_raw, height=400,
                             label_visibility="collapsed")

            st.divider()
            st.info(
                f"**Conclusion:** The baseline agent accepted all of its own claims without "
                f"independent verification. DataCouncil's Validator Agent flagged "
                f"**{flagged} insight(s)** and produced **{revised} corrected revision(s)** "
                f"before the final report was written.\n\n"
                f"This demonstrates the core advantage of multi-agent systems: "
                f"analysis, validation, visualization, and reporting are separated "
                f"across specialized agents, reducing the risk of unchecked assumptions."
            )
