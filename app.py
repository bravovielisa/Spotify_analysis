#-------------------LIBRERIAS-----------------------#
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image #Para poder leer las imagenes jpg
import base64 #Para el gif
import io #Para ver la df.info()
#-------------------LIBRERIAS-----------------------#


#-------------------CONFIGURACIÓN-----------------------#


# Configurar el tema de la aplicación a oscuro


st.set_page_config(
    page_title='¿Cómo te sientes con Spotify?',
    page_icon='🎵',
    layout="wide",  # Puedes ajustar el diseño de la página según tus necesidades
    initial_sidebar_state="expanded",  # Puedes elegir si la barra lateral estará expandida o contraída al inicio
)

# Para que a la gente que use el codigo no le aparezcan los warnings de cambios en las librerias ponemos:
st.set_option('deprecation.showPyplotGlobalUse', False)

#-------------------CONFIGURACIÓN-----------------------#


#-------------------COSAS QUE VAMOS A USAR EN TODA LA APP-----------------------#
# opening the image
image1 = Image.open('img/encabezado.PNG')
image2 = Image.open('img/Happy.PNG')
image3 = Image.open('img/Sad.PNG')
image4 = Image.open('img/Fear.PNG')
image5 = Image.open('img/Anger.PNG')
image6 = Image.open('img/Focus.PNG')



# gif from local file
#Gif Sidebar Menu
file_ = open('img/Spotify.gif', "rb")
contents = file_.read()
data_url1 = base64.b64encode(contents).decode("utf-8")
file_.close()


# Crear elementos en el menú sidebar

# Puedes agregar botones, radio botones u otros componentes aquí
st.sidebar.header('Menú')
st.sidebar.markdown(
        f'<img src="data:image/gif;base64,{data_url1}" alt="cat gif" width="300" height="120">',
        unsafe_allow_html=True,
    ) 
selected_option = st.sidebar.selectbox('Selecciona una opción', ('Inicio','Importación', 'EDA', 'Machine Learning'))
st.sidebar.markdown("</div>", unsafe_allow_html=True)


#Dataframes


#--------------------gráficas----------------------------#



#--------------------gráficas----------------------------#

#--------------------------------------INICIO--------------------------------------#

# Lógica para cada opción seleccionada
if selected_option == 'Inicio':
    # Contenido de la página de inicio:
    #Titulo:
    title_html = """
    <h1 style="color: #191414;">¡Bienvenidos a mi proyecto final!</h1>
"""
    st.markdown(title_html, unsafe_allow_html=True)
    #texto:
    st.markdown("""<span style='text-align: center; color: black;'>Para este proyecto se me ha ocurrido investigar las entrañas de una de mis aplicaciones favoritas: Spotify💚.  
                Como amante de la música que soy me parecía interesante tener la oportunidad de poder navegar en su base de datos.  
                </h2>""", unsafe_allow_html=True)
    st.markdown('''Este proyecto pretende pasar por un proceso de análisis bastante completo, desde la importación del dataset, hasta la elaboración de algoritmos de clasificación y de regresión.   
                ''')
    st.markdown('''<span style='text-align: center; color: green;'>Para entender mejor el proceso, nos centraremos en una serie de playlists seleccionadas personalmente y creadas por Spotify que transmiten o pretenden transmitir sensaciones relacionadas con tu **estado de ánimo**.</h2>''', unsafe_allow_html=True)
    
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write(' ')

    with col2:
        st.image(image1, width=900)

    with col3:
        st.write(' ')
    with col4:
        st.write(' ')
    with col5:
        st.write(' ')
        
    #Titulo:
    title_html1 = """
    <h1 style="color: #1db954;">🎭¿Cómo identifica Spotify los estados de ánimo? 🎭:</h1>
"""
    st.markdown(title_html1, unsafe_allow_html=True)
    st.markdown('')
    
    st.image(image2, width=300)
    
    
    
    
