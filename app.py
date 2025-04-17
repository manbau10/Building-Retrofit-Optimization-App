import streamlit as st
import pandas as pd
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.util.ref_dirs import get_reference_directions
import io

# Initialize session state for page navigation and results persistence
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'optimization_currency' not in st.session_state:
    st.session_state.optimization_currency = None

def set_page(page_name):
    st.session_state.current_page = page_name

# Shared utility functions
def detect_decision_variables(df):
    """Auto-discover variables with limited unique values"""
    candidates = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 20:
            candidates.append(col)
    return candidates

class AutoRetrofitProblem(ElementwiseProblem):
    def __init__(self, df, decision_vars, value_maps, params):
        self.df = df
        self.decision_vars = decision_vars
        self.value_maps = value_maps
        self.params = params
        
        n_var = len(decision_vars)
        xl = [0] * n_var
        xu = [len(values)-1 for values in value_maps.values()]
        
        super().__init__(n_var=n_var, n_obj=3, n_constr=3,
                        xl=np.array(xl), xu=np.array(xu))
        self.PPD_base = params['PPD_BASE']

    def _evaluate(self, x, out, *args, **kwargs):
        conditions = []
        for i, var in enumerate(self.decision_vars):
            option_index = int(x[i])
            value = self.value_maps[var][option_index]
            conditions.append(f"`{var}` == '{value}'")
        
        query = " & ".join(conditions)
        row = self.df.query(query)
        
        if len(row) == 0:
            out["F"] = [1e5, 1e5, 1e5]
            out["G"] = [1e5, 1e5, 1e5]
        else:
            out["F"] = [
                -row['Csave'].values[0],
                -row['EP'].values[0],
                row[self.params['target_cols']['ppd']].values[0]
            ]
            out["G"] = [
                -row['Csave'].values[0],
                -row['EP'].values[0],
                row[self.params['target_cols']['ppd']].values[0] - self.PPD_base
            ]

def run_optimization(df, decision_vars, value_maps, params):
    problem = AutoRetrofitProblem(df, decision_vars, value_maps, params)
    
    algorithm = {
        "NSGA-II": NSGA2(pop_size=100),
        "NSGA-III": NSGA3(ref_dirs=get_reference_directions("das-dennis", 3, n_partitions=12)),
        "AGE-MOEA": AGEMOEA(pop_size=100)
    }[params['algorithm']]
    
    try:
        with st.spinner(f'Running {params["algorithm"]} optimization...'):
            res = minimize(problem, algorithm, ('n_gen', 50), seed=1, verbose=False)
            if res is None:
                st.error("Optimization failed to produce results")
                return None
            results = process_results(res, decision_vars, value_maps, params)
            # Store results in session state
            st.session_state.optimization_results = results
            st.session_state.optimization_currency = params['currency']
            return results
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return None

def process_results(res, decision_vars, value_maps, params):
    if res is None or res.X is None:
        return None
        
    results_df = pd.DataFrame(res.X, columns=decision_vars)
    
    for var in decision_vars:
        results_df[var] = results_df[var].apply(
            lambda x: value_maps[var][int(x)] if x is not None else None
        )
    
    F = res.F
    results_df['Carbon Saving'] = -F[:, 0]
    results_df[f'Economic Profitability ({params["currency"]})'] = -F[:, 1]
    results_df['PPD'] = F[:, 2]
    
    return results_df.sort_values('Carbon Saving', ascending=False)

# Preprocessing functions for different optimization types
def preprocess_data_lca(df, params, decision_vars, target_cols):
    """LCA-based preprocessing"""
    df = df.copy()
    
    # Ensure numeric type for calculations
    for col in target_cols.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert decision variables to strings
    for var in decision_vars:
        df[var] = df[var].astype(str)
    
    # Calculate derived columns for LCA
    df['Csave'] = (params['C_BASE'] - df[target_cols['carbon']]) * params['LT'] - df[target_cols['embodied']]
    df['Esave'] = (params['E_BASE'] - df[target_cols['energy']]) * params['A'] * params['LT']
    df['EP'] = (df['Esave'] * params['EC']) - df[target_cols['cost']]
    
    return df[decision_vars + ['Csave', 'EP', target_cols['ppd']]]

def preprocess_data_annual(df, params, decision_vars, target_cols):
    """Annual-based preprocessing"""
    df = df.copy()
    
    # Ensure numeric type for calculations
    for col in target_cols.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert decision variables to strings
    for var in decision_vars:
        df[var] = df[var].astype(str)
    
    # Calculate derived columns for annual basis
    df['Csave'] = params['C_BASE'] - (df[target_cols['carbon']] + (df[target_cols['embodied']] / params['LT']))
    df['Esave'] = (params['E_BASE'] - df[target_cols['energy']]) * params['A']
    df['EP'] = (df['Esave'] * params['EC']) - (df[target_cols['cost']] / params['LT'])
    
    return df[decision_vars + ['Csave', 'EP', target_cols['ppd']]]

def load_data():
    """Common data loading function"""
    data_source = st.radio(
        "Choose data source",
        ["Use Sample Data", "Upload Own Data"],
        horizontal=True
    )

    if data_source == "Use Sample Data":
        sample_choice = st.selectbox(
            "Select a sample dataset",
            ["Sample 1", "Sample 2", "Sample 3"]
        )
        
        try:
            if sample_choice == "Sample 1":
                df = pd.read_excel("Sample 1.xlsx")
            elif sample_choice == "Sample 2":
                df = pd.read_excel("Sample 2.xlsx")
            else:
                df = pd.read_excel("Sample 3.xlsx")
            
            st.success(f"Loaded {sample_choice} successfully!")
            return df
            
        except Exception as e:
            st.error(f"Error loading {sample_choice}: {str(e)}")
            st.stop()

    else:
        data_file = st.file_uploader("Upload data file", type=["csv", "xlsx"])
        
        if not data_file:
            st.warning("Please upload a data file")
            st.stop()
        
        try:
            if data_file.name.endswith('.csv'):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

def show_visualization(results, currency):
    """Common visualization function"""
    if results is not None and not results.empty:
        unique_solutions = results.drop_duplicates().sort_values('Carbon Saving', ascending=False)
        
        st.header("Optimization Results")
        st.dataframe(unique_solutions, use_container_width=True)
        
        # Pareto Front Visualization
        st.subheader("Pareto Front")
        fig_pareto = plt.figure(figsize=(12, 8))
        ax = fig_pareto.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(unique_solutions['Carbon Saving'],
                           unique_solutions[f'Economic Profitability ({currency})'],
                           unique_solutions['PPD'],
                           c=unique_solutions['PPD'],
                           cmap='viridis')
        
        ax.set_xlabel('Carbon Saving')
        ax.set_ylabel(f'Economic Profitability ({currency})')
        ax.set_zlabel('PPD', labelpad=10)
        plt.colorbar(scatter, label='PPD')
        plt.title('Pareto Front of Solutions')
        
        ax.view_init(elev=20, azim=45)
        plt.tight_layout(pad=2.0)
        fig_pareto.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        
        st.pyplot(fig_pareto)
        
        # Add download buttons and additional visualizations
        pareto_buf = io.BytesIO()
        fig_pareto.savefig(pareto_buf, format='png', dpi=300, bbox_inches='tight')
        pareto_buf.seek(0)  # Important: reset buffer position to beginning
        st.download_button(
            label="Download Pareto Front Plot",
            data=pareto_buf.getvalue(),
            file_name="pareto_front.png",
            mime="image/png",
            key='download-pareto'
        )
        
        # Solution Space Analysis
        st.subheader("Solution Space Analysis")
        fig_space = plt.figure(figsize=(18, 5))
        
        metrics = [
            ('Carbon Saving', f'Economic Profitability ({currency})', 'PPD'),
            ('Carbon Saving', 'PPD', f'Economic Profitability ({currency})'),
            (f'Economic Profitability ({currency})', 'PPD', 'Carbon Saving')
        ]
        
        for i, (x, y, c) in enumerate(metrics):
            ax = fig_space.add_subplot(1, 3, i+1)
            scatter = ax.scatter(unique_solutions[x], unique_solutions[y], 
                               c=unique_solutions[c], cmap='viridis')
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            plt.colorbar(scatter, ax=ax, label=c)
        
        plt.tight_layout()
        st.pyplot(fig_space)
        
        # Add download buttons
        space_buf = io.BytesIO()
        fig_space.savefig(space_buf, format='png', dpi=300, bbox_inches='tight')
        space_buf.seek(0)  # Important: reset buffer position to beginning
        st.download_button(
            label="Download Solution Space Plot",
            data=space_buf.getvalue(),
            file_name="solution_space.png",
            mime="image/png",
            key='download-space'
        )
        
        # Export results
        csv = unique_solutions.to_csv(index=False).encode()
        st.download_button(
            "Download Results CSV",
            csv,
            "optimization_results.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Clear matplotlib figures
        plt.close(fig_pareto)
        plt.close(fig_space)
    else:
        st.error("No valid optimization results found")
        
# Page-specific layouts
def home_page():
    st.title("Building Net Zero Carbon (NZC) Retrofit Optimizer")
    
    st.markdown("""
    Welcome to the Building Net Zero Carbon (NZC) Retrofit Optimizer

    Retrofitting buildings to Net Zero Carbon (NZC) is an efficient way to lower carbon emissions and prevent climate change. However, it is challenging to select the optimal retrofit strategies to maximize social, environmental, and economic sustainability while retrofitting the buildings to NZC. Therefore, this app supports deciding optimal retrofit strategies to attain NZC buildings, focusing on carbon savings, economic profitability, and occupant satisfaction. 

    ### Available Analysis Methods

    1. **Life Cycle Analysis (LCA) based Optimization**
       * Considers the life cycle performance of retrofit measures.
       * Optimization is based on life cycle carbon saving, economic profitability, and occupant dissatisfaction.

    2. **Annual based Optimization**
       * Considers annual performance of retrofit measures
       * Optimization is based on annual carbon saving, economic profitability, and occupant dissatisfaction.

    ### How to Use

    1. Select your desired analysis method from the sidebar
    2. Upload your data (the app can be tested using the sample data from the data selection on the right side of the app)
    3. Configure parameters and run optimization
    4. Analyze results and download reports

    ### Detailed User Manual
    """)

    # Add download button for detailed manual
    try:
        with open("Detailed User Manual.pdf", "rb") as file:
            st.download_button(
                label="Download Detailed User Manual",
                data=file,
                file_name="Detailed User Manual.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"Error loading user manual file: {str(e)}")
    
def optimization_page(optimization_type):
    title = "🔄 Life Cycle Analysis (LCA) Optimization" if optimization_type == "LCA" else "📅 Annual-based Optimization"
    st.title(title)
    
    # Data Selection/Upload
    st.header("Data Selection")
    df = load_data()
    
    if df is None:
        return
    
    # Auto-detect potential decision variables
    decision_candidates = detect_decision_variables(df)
    
    if not decision_candidates:
        st.error("No suitable decision variables found in data")
        return
    
    # Configuration
    st.header("Configuration")
    decision_vars = st.multiselect(
        "Select decision variables",
        decision_candidates,
        default=decision_candidates[:min(5, len(decision_candidates))]
    )
    
    if not decision_vars:
        st.error("Select at least 1 decision variable")
        return
    
    # Target column mapping
    st.subheader("Map Target Columns")
    target_cols = {
        'energy': st.selectbox("Energy Consumption Column", df.columns),
        'carbon': st.selectbox("Carbon Emission Column", df.columns),
        'embodied': st.selectbox("Embodied Carbon Column", df.columns),
        'cost': st.selectbox("Initial Cost Column", df.columns),
        'ppd': st.selectbox("PPD Column", df.columns)
    }
    
    # Parameters
    currency = st.selectbox("Currency Symbol", ["HKD", "USD", "EUR", "GBP", "Other"])
    
    st.sidebar.header("Parameters")
    params = {
        'E_BASE': st.sidebar.number_input(
            "Energy Baseline (kWh/m²)" if optimization_type == "Annual" else "Energy Baseline (kWh/m²)", 
            value=111.4
        ),
        'C_BASE': st.sidebar.number_input(
            "Carbon emissions baseline (kgCO₂)" if optimization_type == "Annual" else "Carbon emissions baseline (kgCO₂)", 
            value=1384.5
        ),
        'PPD_BASE': st.sidebar.number_input("Predicted Percentage of Dissatisfaction (PPD) Baseline", value=14.68),
        'LT': st.sidebar.number_input("Lifetime (years)", value=30),
        'EC': st.sidebar.number_input(f"Energy Cost per 1kWh ({currency})", value=0.918),
        'A': st.sidebar.number_input("Area (m²)", value=23.94),
        'algorithm': st.sidebar.selectbox("Algorithm", ["NSGA-II", "NSGA-III", "AGE-MOEA"]),
        'currency': currency,
        'target_cols': target_cols
    }
    
    # Data processing
    try:
        with st.spinner('Analyzing data...'):
            value_maps = {var: sorted(df[var].astype(str).unique()) for var in decision_vars}
            if optimization_type == "LCA":
                processed_df = preprocess_data_lca(df, params, decision_vars, target_cols)
            else:
                processed_df = preprocess_data_annual(df, params, decision_vars, target_cols)
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return
    
    # Data analysis display
    st.header("Data Analysis")
    st.subheader("Decision Variable Options")
    
    cols = st.columns(len(decision_vars))
    for i, var in enumerate(decision_vars):
        with cols[i]:
            st.metric(
                label=var,
                value=f"{len(value_maps[var])} Options",
                help=f"Values: {', '.join(value_maps[var])}"
            )
    
    st.subheader("Processed Data Preview")
    st.dataframe(processed_df.head(), use_container_width=True)
    
    # Optimization and Visualization
    if st.button("Start Optimization"):
        results = run_optimization(processed_df, decision_vars, value_maps, params)
        show_visualization(results, currency)
    # Display previous results if they exist (even after button clicks)
    elif st.session_state.optimization_results is not None:
        show_visualization(st.session_state.optimization_results, st.session_state.optimization_currency)

def main():
    st.set_page_config(page_title="Building Retrofit Optimizer", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis Method",
        ["Home", "LCA Optimization", "Annual Optimization"]
    )
    
    # Clear results when changing pages
    if ('current_page' in st.session_state and 
        page != st.session_state.current_page):
        st.session_state.optimization_results = None
        st.session_state.optimization_currency = None
        st.session_state.current_page = page
    
    # Display selected page
    if page == "Home":
        home_page()
    elif page == "LCA Optimization":
        optimization_page("LCA")
    else:
        optimization_page("Annual")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        This application is designed to help optimize building retrofit strategies 
        using different analysis approaches. Choose your preferred method from the 
        options above to begin your analysis.
        """
    )

if __name__ == "__main__":
    main()
