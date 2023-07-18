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
    st.markdown('Se han seleccionado las siguientes playlists para el análisis:')
    col1, col2 = st.columns(2)
    with col1:
        st.write('**1) Playlist relacionada con la felicidad. "Mood": "Happy" en el dataframe:**')
        st.image(image2, width=500)
    with col2:
        st.write('**2) Playlist relacionada con la tristeza. "Mood": "Sad" en el dataframe:**')
        st.image(image3, width=500)
    col1, col2 = st.columns(2)
    with col1:
        st.write('**3) Playlist relacionada con el miedo. "Mood": "Fear" en el dataframe:**')
        st.image(image4, width=500)
    with col2:
        st.write('**4) Playlist relacionada con la ira. "Mood": "Anger" en el dataframe:**')
        st.image(image5, width=750)
    st.write('**5) Playlist relacionada con la concentración. "Mood": "Focus"** en el dataframe:')
    st.image(image6, width=500)
    
    #Tabla características de las canciones:
    
    st.subheader("Características de las canciones:")
    st.write('A continuación los parámetros con los que vamos a trabajar de la librería "spotipy" son los siguientes:')
    data = [
        ["Acousticness", "1,0 representa una confianza alta en que la pista es acústica.", "0,0 a 1,0"],
        ["Danceability", "Un valor de 0,0 es el menos bailable y 1,0 el más bailable.", "0,0 a 1,0"],
        ["duration_ms", "Duración de la pista en milisegundos.", "-"],
        ["Energy", "Cuanto más se acerque el valor a 1,0 más enérgica es la canción.", "0,0 a 1,0"],
        ["Instrumentalness", "Cuanto más se acerque el valor a 1,0 mayor será la probabilidad de que la pista no contenga voces.", "0,0 a 1,0"],
        ["Key", "La tonalidad de la pista.", "0 = C, 1 = C♯/D♭, 2 = D, y así sucesivamente. Ninguna clave = -1"],
        ["Liveness", "> 0,8 proporciona una gran probabilidad de que la pista sea en directo.", "0,0 a 1,0"],
        ["Loudness", "La sonoridad global de una pista en decibelios (dB).", "-60 a 0 db"],
        ["Mode", "Mayor se representa con 1 y menor con 0.", "1 o 0"],
        ["Speechiness", "<0,33 canciones no habladas / 0,33 - 0,66 música y voz / >0,66 canciones habladas", "0,0 a 1,0"],
        ["Tempo", "Pulsaciones por minuto (BPM)", "-"],
        ["Valence", "Cuanto más alto más alegre es la pista.", "0,0 a 1,0"],
        ["time_signature", "Compases de \"3/4\" a \"7/4\"", "3 y 7"],
        ["Popularity", "Siendo 100 la canción más popular.", "0 a 100"],
    ]

    # Utilizamos pandas para convertir la lista en un DataFrame
    tabla = pd.DataFrame(data, columns=["Característica", "Descripción", "Valores"])
    styler = tabla.style.hide() #La funcion hide esconde los indices del df

    st.write(styler.to_html(), unsafe_allow_html=True)
    st.write('')
  
   
   
    st.write('''**Danceability**: La bailabilidad describe lo adecuada que es una pista para bailar basándose en una combinación de elementos musicales como el tempo, la estabilidad del ritmo, la fuerza del compás y la regularidad general. Un valor de 0,0 es el menos bailable y 1,0 el más bailable.  
**Energy**: La energía es una medida de 0,0 a 1,0 y representa una medida perceptiva de intensidad y actividad. Normalmente, las pistas energéticas son rápidas, tienen un volumen alto y ruidosas. Por ejemplo, el death metal tiene una energía alta, mientras que un preludio de Bach tiene una puntuación baja en la escala. Las características perceptivas que contribuyen a este atributo incluyen el rango dinámico, el volumen percibido, el timbre, la velocidad de inicio y la entropía general.  
**Instrumentalness**: Predice si una pista no contiene voces. Los sonidos "ooh" y "aah" se consideran instrumentales en este contexto. Las pistas de rap son claramente "vocales". Cuanto más se acerque el valor a 1,0 mayor será la probabilidad de que la pista no contenga voces. Los valores superiores a 0,5 representan pistas instrumentales, pero la confianza es mayor a medida que el valor se acerca a 1,0.  
**Key**: La tonalidad de la pista. Los números enteros se asignan a tonos utilizando la notación estándar Pitch Class. Por ejemplo, 0 = C, 1 = C♯/D♭, 2 = D, y así sucesivamente. Si no se detectó ninguna clave, el valor es -1.  
**Liveness**: Detecta la presencia de público en la grabación. Los valores de liveness más altos representan una mayor probabilidad de que la pista se haya interpretado en directo. Un valor superior a 0,8 proporciona una gran probabilidad de que la pista sea en directo.  
**Loudness**: La sonoridad global de una pista en decibelios (dB). Los valores de sonoridad se promedian en toda la pista. Los valores suelen oscilar entre -60 y 0 db.  
**Mode**: El modo indica la modalidad (mayor o menor) de una pista, el tipo de escala del que se deriva su contenido melódico. Mayor se representa con 1 y menor con 0.  
**Speechiness**: La locuacidad detecta la presencia de palabras habladas en una pista. Cuanto más exclusivamente hablada sea la grabación (por ejemplo, programa de entrevistas, audiolibro, poesía), más se acercará a 1,0 el valor del atributo. Los valores superiores a 0,66 describen pistas que probablemente estén compuestas en su totalidad por palabras habladas. Los valores entre 0,33 y 0,66 describen pistas que pueden contener tanto música como voz, ya sea en secciones o en capas, incluyendo casos como la música rap. Los valores inferiores a 0,33 representan probablemente música y otras pistas no habladas.  
**Tempo**: El tempo global estimado de una pista en pulsaciones por minuto (BPM). En terminología musical, el tempo es la velocidad o el ritmo de una pieza determinada y se deriva directamente de la duración media de los tiempos.  
**Valence**: Medida de 0,0 a 1,0 que describe la positividad musical que transmite una pista. Las pistas con valencia alta suenan más positivas (por ejemplo, felices, alegres, eufóricas), mientras que las pistas con valencia baja suenan más negativas (por ejemplo, tristes, deprimidas, enfadadas).  
**time_signature**: Un compás estimado. El compás es una convención de notación que especifica cuántos tiempos hay en cada compás. El compás oscila entre 3 y 7, lo que indica compases de "3/4" a "7/4".  
**track_href**: Un enlace al terminal de la API web que proporciona todos los detalles de la pista.''')

    st.write('''Librería spotipy: https://spotipy.readthedocs.io/en/2.19.0/  
    Características de las pistas: https://developer.spotify.com/documentation/web-api/reference/get-audio-features''')
