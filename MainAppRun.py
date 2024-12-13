import streamlit as st
import subprocess
import os

# Function to run Streamlit sub-apps
def run_sub_app(file_path):
    if os.path.exists(file_path):
        subprocess.run(["streamlit", "run", file_path])
    else:
        st.error(f"File `{file_path}` does not exist. Please check the file path.")

# Main interface design for `main.py`
st.title("Dataset Analysis and Modeling Dashboard")
st.write("""
    Welcome to the data analysis and modeling application. 
    Please select a category and analysis type from the dropdown menus in the left sidebar, then click **GO** to proceed.
""")
st.write("#### Instructions:")
st.write("""
1. **CO2 Emissions:** Analyze and build models based on CO2 emissions data.
2. **GHG:** Analyze and build models based on greenhouse gas (GHG) data.
3. **Global Temperature:** Analyze and build models based on global temperature data.
""")
st.info("Ensure that sub-files (Explore, Analysis, Train Model) are correctly placed.")

# Sidebar dropdown for main menu
st.sidebar.title("Main Menu")
main_tab = st.sidebar.selectbox(
    "Select Dataset",
    ["Select...", "CO2 Emissions", "GHG", "Global Temperature"]
)

# Conditional dropdown for sub-tabs
sub_tab = None
if main_tab != "Select...":
    st.sidebar.title(f"{main_tab} Menu")
    sub_tab = st.sidebar.selectbox(
        "Select an Analysis Type",
        ["Select...", "Explore", "Analysis", "Train Model"]
    )

# Add a GO button
if main_tab != "Select..." and sub_tab != "Select...":
    if st.sidebar.button("GO"):
        if main_tab == "CO2 Emissions":
            if sub_tab == "Explore":
                st.write("### Exploring CO2 Emissions Data")
                run_sub_app("./CO2 Emissions/UI/streamlit_explore.py")
            elif sub_tab == "Analysis":
                st.write("### Analyzing CO2 Emissions Data")
                run_sub_app("./CO2 Emissions/UI/streamlit_analysis.py")
            elif sub_tab == "Train Model":
                st.write("### Training Model on CO2 Emissions Data")
                run_sub_app("./CO2 Emissions/UI/streamlit_app_final.py")
        
        elif main_tab == "GHG":
            if sub_tab == "Explore":
                st.write("### Exploring GHG Data")
                run_sub_app("./GHG/UI/streamlit_explore.py")
            elif sub_tab == "Analysis":
                st.write("### Analyzing GHG Data")
                run_sub_app("./GHG/UI/streamlit_analysis.py")
            elif sub_tab == "Train Model":
                st.write("### Training Model on GHG Data")
                run_sub_app("./GHG/UI/streamlit_app.py")
        
        elif main_tab == "Global Temperature":
            if sub_tab == "Explore":
                st.write("### Exploring Global Temperature Data")
                run_sub_app("./Global Temperature/UI/streamlit_explore.py")
            elif sub_tab == "Analysis":
                st.write("### Analyzing Global Temperature Data")
                run_sub_app("./Global Temperature/UI/streamlit_analysis.py")
            elif sub_tab == "Train Model":
                st.write("### Training Model on Global Temperature Data")
                run_sub_app("./Global Temperature/UI/streamlit_app_rnn_lstm.py")
    else:
        st.sidebar.info("Click **GO** to proceed.")
