import streamlit as st
import option_models # local library that calculates models 
import option_models_fast # implementation developed using Cython
from tree_pricing_drawings import BinomialTree, TrinomialTree # local tree plotting library 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from enum import Enum
import numpy as np
import time



# Define default parameters
S = 100.
K = 110.
T = 0.5
r = 0.03
sigma = 0.3


class OPTION_PRICING_MODEL(Enum):
    OPTION_PRICING = "Option pricing"
    Tree_plot = "Tree plotter"
    Comparison = "Model Comparison"

class EU_AM(Enum):
    Eur = "European"
    Am = "American"
    
    


descriptions = {
    'Black-Scholes': r"""
    - **Function:** `option_models.black_scholes`
    - **Description:** Black–Scholes prices European call and put options using closed-form formulae.
    - **Inputs:** `(S, K, T, r, sigma)` where `S` is spot, `K` strike, `T` time to maturity, `r` risk-free rate, `sigma` volatility.
    """,
    'Jarrow-Rudd': r"""
    - **Function:** `option_models.jarrow_rudd`
    - **Description:** Jarrow–Rudd binomial tree for option pricing (supports American/European, call/put).
    - **Notes:** Use `n` steps to control accuracy; set `return_trees=True` to retrieve the intermediate trees.
    """,
    'Cox-Ross-Rubinstein': r"""
    - **Function:** `option_models.cox_ross_rubinstein`
    - **Description:** Cox–Ross–Rubinstein (CRR) binomial tree implementation for option pricing.
    - **Notes:** CRR uses symmetric up/down factors `u=exp(sigma*sqrt(h))`, `d=1/u`.
    """,
    'Kamrad-Ritchken': r"""
    - **Function:** `option_models.kamrad_ritchken`
    - **Description:** Kamrad–Ritchken trinomial tree model (supports American/European, call/put).
    - **Notes:** Trinomial trees allocate a wider state-space for better stability with larger steps.
    """
}

# Use explicit figure passing to Streamlit instead of suppressing pyplot warnings

# User selected model from sidebar 
pricing_method = st.sidebar.radio('Section selection:', options=[model.value for model in OPTION_PRICING_MODEL])
Fast_option = st.sidebar.checkbox("Enable Fast calculation with Cython")


st.sidebar.markdown('_'*12)

option_functions_default = {
    'Black-Scholes': option_models.black_scholes,
    'Jarrow-Rudd': option_models.jarrow_rudd,
    'Cox-Ross-Rubinstein': option_models.cox_ross_rubinstein,
    'Kamrad-Ritchken': option_models.kamrad_ritchken
}
option_functions_fast = {
    'Black-Scholes': option_models_fast.black_scholes,
    'Jarrow-Rudd': option_models_fast.jarrow_rudd,
    'Cox-Ross-Rubinstein': option_models_fast.cox_ross_rubinstein,
    'Kamrad-Ritchken': option_models_fast.kamrad_ritchken
}


function_kind={False:option_functions_default,True:option_functions_fast}
module_choice={False:option_models,True:option_functions_fast}


if pricing_method == OPTION_PRICING_MODEL.OPTION_PRICING.value:
    option_functions=function_kind[Fast_option]
    st.title(pricing_method)
    function_choice = st.radio("Select pricing model:", list(option_functions.keys()))
    S = st.sidebar.number_input('Stock Price (S):', value=S)
    K = st.sidebar.number_input('Strike (K):', value=K)
    T = st.sidebar.number_input('Time to Maturity (T):', value=T)
    r = st.sidebar.number_input('Risk-Free Rate (r):', value=r)
    sigma = st.sidebar.number_input('Volatility (sigma):', value=sigma)
    option_call_put = st.sidebar.radio('Please select option kind', options=['Call','Put'])
    cp = 1 if option_call_put == "Call" else -1
    n=None
    am=None
    if function_choice !='Black-Scholes':
        option_type = st.sidebar.radio('Please select option type', options=[op_type.value for op_type in EU_AM])
        n = st.slider('Number of time steps:', min_value=3, value=100, max_value=200)

        am = True if option_type == EU_AM.Am.value else False
    
    if st.button('Calculate Option Price'):
    
        option_price = option_functions[function_choice](s=S, k=K, t=T, v=sigma, rf=r, cp=cp,am=am,n=n)
        st.subheader(f"Le prix de l'option avec le modele {function_choice}  est: {option_price}")
        
    st.markdown(descriptions[function_choice])
    
elif pricing_method == OPTION_PRICING_MODEL.Tree_plot.value:
    option_functions=function_kind[Fast_option]
    function_choice = st.radio("Select pricing model:", ['Jarrow-Rudd', 'Cox-Ross-Rubinstein', 'Kamrad-Ritchken'])
    S = st.sidebar.number_input('Stock Price (S):', value=S,)
    K = st.sidebar.number_input('Strike (K):', value=K)
    T = st.sidebar.number_input('Time to Maturity (T):', value=T)
    r = st.sidebar.number_input('Risk-Free Rate (r):', value=r)
    sigma = st.sidebar.number_input('Volatility (sigma):', value=sigma)
    option_call_put = st.sidebar.radio('Please select option kind', options=['Call','Put'])
    cp = 1 if option_call_put == "Call" else -1
    option_type = st.sidebar.radio('Please select option type', options=[op_type.value for op_type in EU_AM])
    am=True if option_type==EU_AM.Am.value else False
    n = st.sidebar.slider('Number of time steps:', min_value=1, value=5, max_value=30)
    tree_choice = st.radio("Sélectionnez le modèle d'évaluation d'options :", ["option_tree","stock_tree"])
    
    
    
    if st.sidebar.button('Draw tree'):
        with st.spinner('Calculating...'):
            start_time = time.time()
            stock_vals = option_functions[function_choice](s=S, k=K, t=T, v=sigma, rf=r, cp=cp,am=am,n=n, return_trees=True)[tree_choice]
        st.write(f"Calculation time {time.time() - start_time:.2f} s")
        with st.spinner('Plotting...'):
            
            tree = TrinomialTree(stock_vals) if function_choice == "Kamrad-Ritchken" else BinomialTree(stock_vals)
            fig = tree.draw_tree()
            st.pyplot(fig)

elif pricing_method == OPTION_PRICING_MODEL.Comparison.value:
    st.title(pricing_method)
    S = st.sidebar.number_input('Stock Price (S):', value=S,)
    K = st.sidebar.number_input('Strike (K):', value=K)
    T = st.sidebar.number_input('Time to Maturity (T):', value=T)
    r = st.sidebar.number_input('Risk-Free Rate (r):', value=r)
    sigma = st.sidebar.number_input('Volatility (sigma):', value=sigma)
    option_call_put = st.sidebar.radio('Please select option kind', options=['Call','Put'])
    cp = 1 if option_call_put == "Call" else -1

    data_points = st.sidebar.slider('Number of data points:', min_value=3, value=10, max_value=1000 if Fast_option else 100)
    option_type = st.sidebar.radio('Please select option type', options=[op_type.value for op_type in EU_AM])
    am=True if option_type==EU_AM.Am.value else False
    
    n_values = np.linspace(10,500 ,data_points)
    if st.sidebar.button('Show results'):
        with st.spinner('Calculating...'):
            start_time = time.time()
            if not am:
                bs_prices = [option_models.black_scholes(S, K, T, sigma, r, cp) for _ in n_values]
            if Fast_option:
                jr_prices = [option_models_fast.jarrow_rudd(S, K, T, sigma, r, cp, n=int(n)) for n in n_values]
                crr_prices = [option_models_fast.cox_ross_rubinstein(S, K, T, sigma, r, cp, n=int(n)) for n in n_values]
                kr_prices = [option_models_fast.kamrad_ritchken(S, K, T, sigma, r, cp, n=int(n)) for n in n_values]
            else:
                jr_prices = [option_models.jarrow_rudd(S, K, T, sigma, r, cp,am=am, n=int(n)) for n in n_values]
                crr_prices = [option_models.cox_ross_rubinstein(S, K, T, sigma, r, cp,am=am, n=int(n)) for n in n_values]
                kr_prices = [option_models.kamrad_ritchken(S, K, T, sigma, r, cp,am=am, n=int(n)) for n in n_values]
        st.write(f"Calculation time {time.time() - start_time:.2f} s")
            
        with st.spinner('Plotting...'):
            fig = go.Figure()   
            if not am:
                fig.add_trace(go.Scatter(x=n_values, y=bs_prices, mode='lines', name='Black-Scholes', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=n_values, y=jr_prices, mode='lines+markers', name='Jarrow-Rudd', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=n_values, y=crr_prices, mode='lines+markers', name='Cox-Ross-Rubinstein', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=n_values, y=kr_prices, mode='lines+markers', name='Kamrad-Ritchken', line=dict(color='purple')))

            # Update x and y-axis labels
            fig.update_xaxes(title='n (Number of Steps)')
            fig.update_yaxes(title='Option Price')

            # Update layout
            fig.update_layout(title=f'Option Pricing Model Convergence Comparison', legend=dict(x=0, y=1, traceorder='normal'),    width=900, height=700 )

            st.plotly_chart(fig)
            if not am:
                # Display the plot
                jr_err=np.abs(np.array(jr_prices)-np.array(bs_prices))
                crr_err=np.abs(np.array(crr_prices)-np.array(bs_prices))
                kr_err=np.abs(np.array(kr_prices)-np.array(bs_prices))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=n_values, y=jr_err, mode='lines+markers', name='Jarrow-Rudd', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=n_values, y=crr_err, mode='lines+markers', name='Cox-Ross-Rubinstein', line=dict(color='green')))    
                fig.add_trace(go.Scatter(x=n_values, y=kr_err, mode='lines+markers', name='Kamrad-Ritchken', line=dict(color='purple')))

                # Update x and y-axis labels
                fig.update_xaxes(title='n (Number of Steps)')
                fig.update_yaxes(title='model Error')

                # Update layout
                fig.update_layout(title=f'Option Pricing Model Convergence Error', legend=dict(x=0, y=1, traceorder='normal'),    width=900, height=700 )

                # Display the plot
                st.plotly_chart(fig)
    
            
        
