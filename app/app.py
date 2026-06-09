# ==========================================
# 1. DJANGO ENVIRONMENT SETUP
# ==========================================
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

# Import your Django models (Update app and model name)
# from my_app.models import VectorData 

# ==========================================
# 2. STREAMLIT APP
# ==========================================
import streamlit as st
st.session_state.clear()
import sopy as sp
import numpy as np
import pandas as pd
from wrapper import SoPyWrapper

st.set_page_config(page_title="SoPy Visualizer", layout="wide")
st.title("SoPy: Tuning & Dynamic Rank Reduction")

# --- INITIALIZATION ---
if 'lattices' not in st.session_state:
    st.session_state.lattices = HeleusLoader().reconstruct(db_entry=SoPyContent.objects.get(uid='c0a2b7bcfbd45293223971021f8127e47a0c570b7a55181606a914876939248b')).lattices()

if 'wrapper' not in st.session_state:
    base_vector = sp.Vector()
    for _ in range(10):
        pos_x, pos_y, pos_z = np.random.uniform(-5, 5, 3)
        base_vector += sp.Vector().gaussian(a=1, positions=[pos_x, pos_y, pos_z], sigmas=[1,1,1], ls=[0,0,0], lattices=st.session_state.lattices)
    st.session_state.wrapper = SoPyWrapper(base_vector)
    # Added 'rank_count' to history tracking
    st.session_state.history = {'step': [], 'update_loss': [], 'tuned_distance': [], 'rank_count': []}
    st.session_state.step_counter = 0

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Learning Parameters")
    learn_rate = st.slider("Learn Rate (alpha)", 0.0, 1.0, 0.05, 0.01)
    learning_tolerance = st.slider("Learning Tolerance", 0.0, 5.0, 0.5, 0.1)
    
    st.header("Tuning Parameters")
    ambiguity_rate = st.slider("Ambiguity Rate (Target Dist)", 0.001, 0.1, 0.01, 0.001)
    tune_rate = st.slider("Tune Step Size", 0.1, 1.0, 0.5, 0.1)
    
    st.header("Dynamic Reduction")
    auto_reduce = st.checkbox("Auto-Reduce Rank", value=True)
    max_allowed_rank = st.number_input("Trigger Reduction at Rank", min_value=2, value=15)
    target_reduction_distance = st.slider("Max Reduction Error (Loss)", 0.001, 0.1, 0.5, 0.001)
    
    st.header("Execution")
    steps_to_run = st.number_input("Steps per click", 1, 50, 5)
    run_cycle = st.button("Run Cycle", type="primary")
    reset_state = st.button("Reset State")

if reset_state:
    st.session_state.clear()
    st.rerun()

# --- DATA INGESTION ---
def get_external_vector():
    return HeleusLoader().reconstruct(db_entry=SoPyContent.objects.get(uid='c0a2b7bcfbd45293223971021f8127e47a0c570b7a55181606a914876939248b'))

# --- EXECUTION LOOP ---
if run_cycle:
    with st.spinner('Processing vectors and dynamically tuning ranks...'):
        for _ in range(steps_to_run):
            st.session_state.step_counter += 1
            input_vec = get_external_vector()
            
            if input_vec is None:
                st.warning("No more vectors available.")
                break
            
            # 1. Update (with tolerance early-stopping)
            loss, did_learn = st.session_state.wrapper.update(input_vec, learn_rate, tolerance=learning_tolerance)
            
            # 2. Tune (towards ambiguity rate)
            dist = st.session_state.wrapper.tune(input_vec, ambiguity_rate, tune_rate).numpy()
            
            # 3. Dynamic Rank Reduction
            current_rank = len(st.session_state.wrapper.vector)
            if auto_reduce and current_rank >= max_allowed_rank:
                new_rank, reduction_error = st.session_state.wrapper.reduce_to_target_distance(
                    max_allowed_distance=target_reduction_distance
                )
                current_rank = new_rank
                st.toast(f"Reduced to rank {new_rank} with error {reduction_error:.4f}")
            
            # Log history
            st.session_state.history['step'].append(st.session_state.step_counter)
            st.session_state.history['update_loss'].append(loss)
            st.session_state.history['tuned_distance'].append(dist)
            st.session_state.history['rank_count'].append(current_rank)

# --- VISUALIZATION ---
if st.session_state.history['step']:
    df = pd.DataFrame(st.session_state.history).set_index('step').sort_index()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Update Loss")
        st.line_chart(df['update_loss'], color="#FF4B4B")
        
    with col2:
        st.subheader("Tuned Ambiguity Distance")
        st.line_chart(df[['tuned_distance']], color=["#1f77b4"])
        
    with col3:
        st.subheader("Canonical SoP Ranks")
        st.line_chart(df['rank_count'], color="#2ca02c")
else:
    st.info("Awaiting execution. Connect your Django data and hit 'Run Cycle'.")
    