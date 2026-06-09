import streamlit as st
import pandas as pd
import numpy as np
import sopy as sp

import os
import sys
import django
from pathlib import Path

# Adjust this path to point to your Django project root
#BASE_DIR = Path(__file__).resolve().parent.parent 
#sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skopos.settings') # Update this!
django.setup()

from heleus.models import SoPyContent # Replace with your app and class name
from heleus.loader import HeleusLoader

# --- Your Class Definition ---
class SoPyWrapper:
    def __init__(self, initial_vector=None):


        base_vector = sp.Vector()
        for _ in range(10):
            pos_x, pos_y, pos_z = np.random.uniform(-5, 5, 3)
            base_vector += sp.Vector().gaussian(a=1, positions=[pos_x, pos_y, pos_z], sigmas=[1,1,1], ls=[0,0,0], lattices=st.session_state.lattices)

        self.vector = base_vector if base_vector is not None else sp.Vector()

    def update(self, input_vector, learn_rate, tolerance=0.0, iterate=1):
        current_dist = self.vector.dist(input_vector)
        if current_dist <= tolerance:
            return current_dist, False 
            
        self.vector = self.vector.learn(input_vector, alpha=learn_rate/100, iterate=iterate)
        return self.vector.dist(input_vector), True

    def tune(self, input_vector, ambiguity_rate, tune_rate=0.01):
        current_dist = self.vector.dist(input_vector)
        dist_error = current_dist - ambiguity_rate
        effective_alpha = tune_rate * dist_error
        
        self.vector = self.vector.learn(input_vector, alpha=abs(effective_alpha), iterate=1)
        return effective_alpha, self.vector.dist(input_vector)

    def reduce_to_target_distance(self, alpha,  max_allowed_distance, iterations=10):
        current_rank = len(self.vector)
        if current_rank <= 1:
            return current_rank, 0.0

        original_vector = self.vector 
        low = 1
        high = current_rank
        best_partition = current_rank
        best_reduced_vector = self.vector
        final_dist = 0.0
        
        while low <= high:
            mid_partition = (low + high) // 2
            test_vector = original_vector.fibonacci(
                partition=mid_partition, 
                iterate=iterations, 
                alpha = alpha,
                total_alpha = alpha,
                total_iterate=iterations
            )
            current_dist = test_vector.dist(original_vector)
            
            if current_dist <= max_allowed_distance:
                best_partition = mid_partition
                best_reduced_vector = test_vector
                final_dist = current_dist
                high = mid_partition - 1
            else:
                low = mid_partition + 1
                
        self.vector = best_reduced_vector
        return best_partition, final_dist

# --- Streamlit UI and Logic ---
st.set_page_config(layout="wide")
st.title("Ambiguity Rate Investigation Window")
st.write("Sweeping ambiguity values to analyze vector distance, reduction error, and partition rank.")
# --- INITIALIZATION ---
if 'lattices' not in st.session_state:
    st.session_state.lattices = HeleusLoader().reconstruct(db_entry=SoPyContent.objects.get(uid='c0a2b7bcfbd45293223971021f8127e47a0c570b7a55181606a914876939248b')).lattices()

# 1. Sidebar Parameter Setup
st.sidebar.header("Static Parameters")
learn_rate = st.sidebar.number_input("Learn Rate", value=0.05)
tune_rate = st.sidebar.number_input("Tune Rate", value=0.5)
max_allowed_dist = st.sidebar.number_input("Max Allowed Distance (Reduction)", value=0.001)

st.sidebar.header("Ambiguity Sweep Range")
amb_min = st.sidebar.number_input("Min Ambiguity", value=0.0)
amb_max = st.sidebar.number_input("Max Ambiguity", value=1.0)
amb_steps = st.sidebar.slider("Number of Data Points", min_value=10, max_value=200, value=10)

# Helper function to prevent Tensor Hash Errors in Pandas/Streamlit
def safe_extract(val):
    return float(val.numpy()) if hasattr(val, 'numpy') else float(val)

# 2. Execution Loop
if st.button("Run Sweep", type="primary"):
    ambiguity_rates = np.linspace(amb_min, amb_max, amb_steps)
    results = []
    
    # Progress indicator
    progress_text = "Running parameter sweep..."
    my_bar = st.progress(0, text=progress_text)
    input_vec = HeleusLoader().reconstruct(db_entry=SoPyContent.objects.get(uid='c0a2b7bcfbd45293223971021f8127e47a0c570b7a55181606a914876939248b'))
    for i, amb_rate in enumerate(ambiguity_rates):
        
        wrapper = SoPyWrapper()
        reduction_error = np.nan
        new_rank = len(wrapper.vector)
        for ic in range(15):

            # Initialize fresh wrapper and a target input vector
            
            # Execute your sequence
            wrapper.update(input_vec, learn_rate=learn_rate)
            alpha, dist_after_tune = wrapper.tune(input_vec, ambiguity_rate=amb_rate, tune_rate=tune_rate)
            
            if ic in [14]: # Midway through, trigger reduction if rank is above threshold
                new_rank, reduction_error = wrapper.reduce_to_target_distance(alpha, max_allowed_distance=max_allowed_dist)
            
                # Log safe float values
                results.append({
                    "Ambiguity Target": safe_extract(amb_rate),
                    "Distance After Tune": safe_extract(dist_after_tune),
                    "Reduction Error": safe_extract(reduction_error), # Normalize error by rank
                    "Final Partition Rank": safe_extract(new_rank)
                })
            
            # Update progress bar
            my_bar.progress((i + 1) / amb_steps, text=f"Processing Ambiguity: {amb_rate:.2f}")
        
    my_bar.empty() # Clear progress bar when done
    
    # 3. Visualization
    st.success("Sweep Complete!")
    df_results = pd.DataFrame(results)
    
    # Create layout columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ambiguity vs. Tune Distance")
        # Set index to Ambiguity Target so it acts as the X-axis
        chart_data_1 = df_results.set_index("Ambiguity Target")["Distance After Tune"].sort_index()
        st.line_chart(chart_data_1, color="#ffaa00")
        
        st.subheader("Ambiguity vs. Final Partition Rank")
        chart_data_3 = df_results.set_index("Ambiguity Target")["Final Partition Rank"].sort_index()    
        st.line_chart(chart_data_3, color="#00ffaa")

    with col2:
        st.subheader("Ambiguity vs. Reduction Error")
        chart_data_2 = df_results.set_index("Ambiguity Target")["Reduction Error"].sort_index()
        st.line_chart(chart_data_2, color="#00aaff")
        
        st.subheader("Raw Data")
        st.dataframe(df_results, use_container_width=True)