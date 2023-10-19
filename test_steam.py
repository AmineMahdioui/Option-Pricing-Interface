import streamlit as st
import option_models # local library that calculates models 
import option_models_fast # implementation developed using Cython
from tree_pricing_drawings import BinomialTree, TrinomialTree # local tree plotting library 
import matplotlib.pyplot as plt 
import networkx as nx
import plotly.graph_objects as go
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import time



# Define default parameters
S = 100.
K = 110.
T = 0.5
r = 0.03
sigma = 0.3


class OPTION_PRICING_MODEL(Enum):
    OPTION_PRICING=" Option pricing"
    Tree_plot="Tree plotter"
    Comparison="Model Comparison"

class EU_AM(Enum):
    Eur="Option European"
    Am="Option American"
    
    


descriptions = {
    'Black-Scholes': r"""
    - **Fonction :** `option_models.black_scholes`
   - **Description :** Le modèle de Black-Scholes est un modèle mathématique utilisé pour estimer le prix des options européennes (c'est-à-dire celles qui ne peuvent être exercées qu'à l'expiration). Il a été développé par Fischer Black et Myron Scholes en 1973. Le modèle de Black-Scholes prend en compte le prix actuel du sous-jacent ($S$), le prix d'exercice de l'option ($K$), le temps restant avant l'expiration ($T$), le taux d'intérêt sans risque ($r$) et la volatilité ($\sigma$) du sous-jacent.
   - **Formule :** La formule de Black-Scholes pour une option d'achat (call) est la suivante :

     $$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$

     Pour une option de vente (put), la formule est :

     $$P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$

     Où :

     $d_1 = \frac{{\ln(S/K) + (r + (\sigma^2)/2)T}}{{\sigma \sqrt{T}}}$

     $d_2 = d_1 - \sigma \sqrt{T}$

     $N(x)$ est la fonction de distribution cumulative de la loi normale standard.
    """,
    'Jarrow-Rudd': r"""
    La fonction `jarrow_rudd` permet d'évaluer le prix d'une option en utilisant le modèle binomial de Jarrow-Rudd.

Paramètres :
- `s` : Prix initial de l'action
- `k` : Prix d'exercice de l'option
- `t` : Temps d'expiration
- `v` : Volatilité
- `rf` : Taux sans risque
- `cp` : +1/-1 pour une option d'achat/vente
- `am` : True/False pour une option américaine/européenne
- `n` : Nombre d'étapes binomiales

Calculs de base :
- $ h = \frac{t}{n} $
- $ u = e^{(rf - 0.5 \cdot v^2)h + v\sqrt{h}} $
- $ d = e^{(rf - 0.5 \cdot v^2)h - v\sqrt{h}} $
- $ R = e^{r_fh} $
- $ q = \frac{R - d}{u - d} $

Calculs des prix de l'actif et de l'option à l'échéance :
- $ \text{stkval}[i, j] = s \cdot u^{(i-j)} \cdot d^j $

Récursion inverse pour le prix de l'option :
- $ \text{optval}[n, j] = \max(0, cp \cdot (\text{stkval}[n, j] - k)) $
- $ \text{optval}[i, j] = \frac{q \cdot \text{optval}[i+1, j] + (1-q) \cdot \text{optval}[i+1, j+1]}{R} $

Si `am` est True, alors :
- $ \text{optval}[i, j] = \max(\text{optval}[i, j], cp \cdot (\text{stkval}[i, j] - k)) $

Si `return_trees` est True, la fonction retourne les arbres de valeurs de l'actif et de l'option. Sinon, elle renvoie le prix de l'option.    
    """,
    'Cox-Ross-Rubinstein': r"""
    
    La fonction `jarrow_rudd` permet d'évaluer le prix d'une option en utilisant le modèle binomial de Jarrow-Rudd.

Paramètres :
- `s` : Prix initial de l'action
- `k` : Prix d'exercice de l'option
- `t` : Temps d'expiration
- `v` : Volatilité
- `rf` : Taux sans risque
- `cp` : +1/-1 pour une option d'achat/vente
- `am` : True/False pour une option américaine/européenne
- `n` : Nombre d'étapes binomiales

Calculs de base :
- $ h = \frac{t}{n} $
- $ u = e^{+ v\sqrt{h}} $
- $ d = e^{- v\sqrt{h}} $
- $ R = e^{r_fh} $
- $ q = \frac{R - d}{u - d} $

Calculs des prix de l'actif et de l'option à l'échéance :
- $ \text{stkval}[i, j] = s \cdot u^{(i-j)} \cdot d^j $

Récursion inverse pour le prix de l'option :
- $ \text{optval}[n, j] = \max(0, cp \cdot (\text{stkval}[n, j] - k)) $
- $ \text{optval}[i, j] = \frac{q \cdot \text{optval}[i+1, j] + (1-q) \cdot \text{optval}[i+1, j+1]}{R} $

Si `am` est True, alors :
- $ \text{optval}[i, j] = \max(\text{optval}[i, j], cp \cdot (\text{stkval}[i, j] - k)) $

Si `return_trees` est True, la fonction retourne les arbres de valeurs de l'actif et de l'option. Sinon, elle renvoie le prix de l'option.

     
    """,
    'Kamrad-Ritchken': r"""
   La fonction `kamrad_ritchken` permet d'évaluer le prix d'une option en utilisant le modèle trinomial de Kamrad-Ritchken.

Paramètres :
- `s` : Prix initial de l'action
- `k` : Prix d'exercice de l'option
- `t` : Temps d'expiration
- `v` : Volatilité
- `rf` : Taux sans risque
- `cp` : +1/-1 pour une option d'achat/vente
- `am` : True/False pour une option américaine/européenne
- `n` : Nombre d'étapes trinomiales

Calculs de base :
- $ h = \frac{t}{n} $
- $ \mu = rf - \frac{v^2}{2} $
- $ \lambda = \sqrt{\frac{1}{1-\frac{1}{3}}} $
- $ \alpha = \sqrt{1 + h \left(\frac{\mu}{v}\right)^2} $
- $ u = e^{v \alpha \lambda \sqrt{h}} $
- $ d = \frac{1}{u} $
- $ R = e^{r_fh} $
- $ pu = \frac{1}{2\lambda^2} + \frac{\mu \sqrt{h}}{2\lambda v} $
- $ pd = \frac{1}{2\lambda^2} - \frac{\mu \sqrt{h}}{2\lambda v} $
- $ pm = 1 - \frac{1}{\lambda^2} $

Calculs des prix de l'actif et de l'option à l'échéance :
- $ \text{stkval}[i, j] = s \cdot u^{(i-j \text{ if } (i \geq j) \text{ else } 0)} \cdot d^{(j-i \text{ if } (i < j) \text{ else } 0)} $

Récursion inverse pour le prix de l'option :
- $ \text{optval}[n, j] = \max(0, cp \cdot (\text{stkval}[n, j] - k)) $
- $ \text{optval}[i, j] = \frac{pu \cdot \text{optval}[i+1, j] + pm \cdot \text{optval}[i+1, j+1] + pd \cdot \text{optval}[i+1, j+2]}{R} $

Si `am` est True, alors :
- $ \text{optval}[i, j] = \max(\text{optval}[i, j], cp \cdot (\text{stkval}[i, j] - k)) $

Si `return_trees` est True, la fonction retourne les arbres de valeurs de l'actif et de l'option. Sinon, elle renvoie le prix de l'option.
    """
}

# Ignore the Streamlit warning for using st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

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
    function_choice = st.radio("Sélectionnez le modèle d'évaluation d'options :", list(option_functions.keys()))
    S = st.sidebar.number_input('Stock Price (S):', value=S,)
    K = st.sidebar.number_input('Strike (K):', value=K)
    T = st.sidebar.number_input('Time to Maturity (T):', value=T)
    r = st.sidebar.number_input('Risk-Free Rate (r):', value=r)
    sigma = st.sidebar.number_input('Volatility (sigma):', value=sigma)
    option_call_put = st.sidebar.radio('Please select option kind', options=['Call','Put'])
    cp= 1 if option_call_put=="Call" else -1
    n=None
    am=None
    if function_choice !='Black-Scholes':
        option_type = st.sidebar.radio('Please select option type', options=[op_type.value for op_type in EU_AM])
        n = st.slider('Number of times teps:', min_value=3,value=100,max_value=200)

        am=True if option_type==EU_AM.Am.value else False
    
    if st.button('Calculate Option Price'):
    
        option_price = option_functions[function_choice](s=S, k=K, t=T, v=sigma, rf=r, cp=cp,am=am,n=n)
        st.subheader(f"Le prix de l'option avec le modele {function_choice}  est: {option_price}")
        
    st.markdown(descriptions[function_choice])
    
elif pricing_method == OPTION_PRICING_MODEL.Tree_plot.value:
    option_functions=function_kind[Fast_option]
    function_choice = st.radio("Sélectionnez le modèle d'évaluation d'options :",['Jarrow-Rudd','Cox-Ross-Rubinstein','Kamrad-Ritchken'])
    S = st.sidebar.number_input('Stock Price (S):', value=S,)
    K = st.sidebar.number_input('Strike (K):', value=K)
    T = st.sidebar.number_input('Time to Maturity (T):', value=T)
    r = st.sidebar.number_input('Risk-Free Rate (r):', value=r)
    sigma = st.sidebar.number_input('Volatility (sigma):', value=sigma)
    option_call_put = st.sidebar.radio('Please select option kind', options=['Call','Put'])
    cp= 1 if option_call_put=="Call" else -1
    option_type = st.sidebar.radio('Please select option type', options=[op_type.value for op_type in EU_AM])
    am=True if option_type==EU_AM.Am.value else False
    n = st.sidebar.slider('Number of times teps:', min_value=1,value=5,max_value=30 )
    tree_choice = st.radio("Sélectionnez le modèle d'évaluation d'options :", ["option_tree","stock_tree"])
    
    
    
    if st.sidebar.button('Draw tree'):
        with st.spinner('Calculating...'):
            start_time = time.time()
            stock_vals = option_functions[function_choice](s=S, k=K, t=T, v=sigma, rf=r, cp=cp,am=am,n=n, return_trees=True)[tree_choice]
        st.write(f"Calculation time {time.time() - start_time:.2f} s")
        with st.spinner('Plotting...'):
            
            tree = TrinomialTree(stock_vals) if function_choice=="Kamrad-Ritchken" else BinomialTree(stock_vals)
            tree.draw_tree()
            st.pyplot()

elif pricing_method == OPTION_PRICING_MODEL.Comparison.value:
    st.title(pricing_method)
    S = st.sidebar.number_input('Stock Price (S):', value=S,)
    K = st.sidebar.number_input('Strike (K):', value=K)
    T = st.sidebar.number_input('Time to Maturity (T):', value=T)
    r = st.sidebar.number_input('Risk-Free Rate (r):', value=r)
    sigma = st.sidebar.number_input('Volatility (sigma):', value=sigma)
    # option_call_put = st.sidebar.radio('Please select option kind', options=['Call','Put'])
    cp= 1 

    data_points=st.sidebar.slider('Number of data points:', min_value=3,value=10,max_value=1000 if Fast_option else 100)
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
    
            
        
