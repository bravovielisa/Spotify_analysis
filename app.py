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
    page_title='Moods de Spotify',
    page_icon='🎵',
    layout="wide",  # Puedes ajustar el diseño de la página según tus necesidades
    initial_sidebar_state="expanded",  # Puedes elegir si la barra lateral estará expandida o contraída al inicio
)

# Para que a la gente que use el codigo no le aparezcan los warnings de cambios en las librerias ponemos:
st.set_option('deprecation.showPyplotGlobalUse', False)

#-------------------CONFIGURACIÓN-----------------------#


#-------------------COSAS QUE VAMOS A USAR EN TODA LA APP-----------------------#
# opening the image
#------Inicio-------#
image1 = Image.open('img/encabezado.PNG')
image2 = Image.open('img/Happy.PNG')
image3 = Image.open('img/Sad.PNG')
image4 = Image.open('img/Fear.PNG')
image5 = Image.open('img/Anger.PNG')
image6 = Image.open('img/Focus.PNG')
image7 = Image.open('img/Keys.PNG')
#------Inicio-------#


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
selected_option = st.sidebar.selectbox('Selecciona una opción', ('Inicio','Importación y preprocesamiento', 'EDA', 'Machine Learning'))
st.sidebar.markdown("</div>", unsafe_allow_html=True)


#Dataframes
df = pd.read_csv('data/datos.csv')
df1 = pd.read_csv('data/no_lej_data.csv')

#--------------------gráficas----------------------------#
# Nulos:
fig, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='Set2', ax=ax)
# Categoricos circular:
mood_values = df['Mood'].value_counts()
fig1 = px.pie(mood_values, values=mood_values.values, names=mood_values.index, title='Distribución tamaño de la muestra:',template='plotly_white')

# Categoricos barras:
top_10_artists = df['artist_name'].value_counts().head(10)
# Agrupar los datos por 'artist_name' y 'Mood' y contar las apariciones
grouped_df = df.groupby(['artist_name', 'Mood']).size().reset_index(name='count')
grouped_df = grouped_df[grouped_df['artist_name'].isin(top_10_artists.index)]
# Crear el gráfico de barras apiladas utilizando Plotly Express
fig2 = px.bar(grouped_df, x='artist_name', y='count', color='Mood', barmode='stack', template='plotly_white',
             labels={'artist_name': 'Artista', 'count': 'Nº de veces que aparece'},
             title='Top 10 artistas más repetidos por estado de ánimo')
fig2.update_layout(xaxis={'categoryorder':'total descending'}) #Así ordenamos el grafico de barras
#Correlaciones
plt.figure(figsize=(10, 8))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df1.select_dtypes(include=[np.number]).corr(), dtype=bool))
heatmap = sns.heatmap(df1.select_dtypes(include=[np.number]).corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='PiYG')
heatmap.set_title('Correlaciones', fontdict={'fontsize':18}, pad=16)


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
    st.markdown('''Este proyecto pretende analizar, a través de Python, diferentes fases relacionadas con el análisis y la ciencia de datos, desde la importación del dataset, hasta la elaboración de algoritmos de clasificación y de regresión.   
                ''')
    st.write('''No te olvides de echarle un vistazo al código 👀.  
             Encontrarás cada fase explicada en los "jupyter notebooks" dentro de la carpeta de Notebooks: https://github.com/bravovielisa/Spotify_analysis''')
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
    st.markdown('Las siguientes playlists han sido seleccionadas con intención de analizar sus características según el estado de ánimo que representan para así poder entrenar un modelo que recomiende una canción según tu estado de ánimo:')
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
    st.write('A continuación los parámetros o características de las canciones, importadas a través de la propia librería de Spotify, spotipy, con los que vamos a trabajar son los siguientes:')
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
  
   
    st.write('Por si caben alguna duda, a continuación definimos en más detalle alguna de las variables que aparecen en la tabla y añadimos alguna más descriptiva:')
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
**track_href**: Un enlace al terminal de la API web que proporciona todos los detalles de la pista.  
**song_uri**: Un enlace al terminal de la API web que proporciona acceso a la pista.''')
    st.write('En cuanto a la tonalidad podemos consultar la siguiente página para entenderlo mejor https://en.wikipedia.org/wiki/Pitch_class. No obstante en esta tabla se resume a que corresponde cada valor de la Pitch Class:')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('')
    with col2:    
        st.image(image7, width=300)
    with col3:
        st.write('')
   
    st.write('''Librería spotipy: https://spotipy.readthedocs.io/en/2.19.0/  
    Características de las pistas: https://developer.spotify.com/documentation/web-api/reference/get-audio-features''')

#--------------------------------------INICIO--------------------------------------#


#--------------------------------------Importación--------------------------------------#
if selected_option == 'Importación y preprocesamiento':
    # Contenido de la página de inicio:
    #Titulo:
     #Titulo:
    tab1, tab2=st.tabs(["**Importación**", "**Preprocesamiento**"])
    with tab1: 
        title_html2 = """
        <h1 style="color: #1db954;">Acceso al dataset 👩‍💻:</h1>"""
        st.markdown(title_html2, unsafe_allow_html=True)
        
        st.write('**Spotipy** es una biblioteca de Python que permite interactuar con la **API de Spotify**. Se utiliza para acceder y manipular datos de Spotify, como obtener información de canciones, artistas, álbumes, listas de reproducción y realizar acciones como reproducir pistas, crear listas de reproducción y mucho más.')
        st.write('Para acceder a los datos de Spotify necesitamos leer las credenciales que nos da la API: clientid y client_secret en su web https://developer.spotify.com/dashboard (después de haber dado de alta nuestra app).')
        code1='''import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials

        with open("api.txt") as f:
                secret_ls = f.readlines()
                client_id = secret_ls[0][:-1]
                secret = secret_ls[1]
                
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=secret)
        sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)'''
        st.code(code1, language='python')
        st.write('Las credenciales como se puede ver en el código están guardados en un archivo .txt')
        st.write('Para seguir viendo como se importa cada uno de los parámetros de las playlists seleccionadas ir a: https://github.com/bravovielisa/Spotify_analysis/blob/main/Notebooks/Import_data.ipynb')
        
    with tab2:
        title_html3 = """
        <h1 style="color: #1db954;">Preprocesamiento 📐:</h1>"""
        st.markdown(title_html3, unsafe_allow_html=True)
        
        st.subheader('Modificación de valores de la columna playlist_name por estado de ánimo:')
        st.write('Uno de los parámetros que hemos importado de la libreria es el nombre de la playlist como "playlist_name" vamos a renombrar esta variable para que indique el estado de ánimo al que hace referencia, por eso la vamos a denomminar "Mood" y a cada playlist le vamos a asignar un estado de ánimo concreto:')
        
        code2 ='''# Cambiamos los valores de la columna de playlist_name por cada estado de ánimo:
        df.loc[df['playlist_name'] == 'Happy Hits!', 'playlist_name'] = 'Happy'
        df.loc[df['playlist_name'] == 'Heartache', 'playlist_name'] = 'Sad'
        df.loc[df['playlist_name'] == 'Walk Like A Badass', 'playlist_name'] = 'Anger'
        df.loc[df['playlist_name'] == 'Spooky', 'playlist_name'] = 'Fear'
        df.loc[df['playlist_name'] == 'Deep Focus', 'playlist_name'] = 'Focus'
        # Cambiamos el nombre de la columna por Mood o estado de ánimo:
        df = df.rename({'playlist_name': 'Mood'}, axis=1)'''
        st.code(code2, language='python')
        st.write('El dataframe quedaría de la siguiente forma:')
        # df.info----
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        # df.info----
        st.write('El tipo de dato en principio es correcto y no queremos modificar ninguno.')
        
        st.subheader('Valores nulos:')
        code3 = '''df.isnull().sum().sum()'''
        st.code(code3, language='python')
        
        st.pyplot(fig)
        
        st.write('En este caso **no hay valores nulos**')
        
        st.subheader('Valores duplicados:')
        code4 = '''df.duplicated().sum()'''
        st.code(code4, language='python')
        st.write('En este caso **no hay valores duplicados**')
        
        st.subheader('Exportación a csv:')
        code5 = '''df.to_csv('datos.csv', index=False)'''
        st.code(code5, language='python')
#--------------------------------------Importación--------------------------------------#        
#--------------------------------------EDA--------------------------------------#       
if selected_option == 'EDA':
    title_html4 = """
            <h1 style="color: #1db954;">EDA (exploratory data analysis) 🔬:</h1>"""
    st.markdown(title_html4, unsafe_allow_html=True)
    
    page_names=['🗃️ Variables categóricas', '🔢 Variables numéricas']
    page= st.radio('¿Qué quieres analizar?',page_names)
    
    if page =='🗃️ Variables categóricas':               
        st.subheader('Análisis variables categóricas:')
        st.write('Tras importar el archivo "data" creado en el apartado de preprocesamiento, vemos que contamos con 465 datos distribuidos de la siguiente manera:')
        st.plotly_chart(fig1)
        st.markdown('''Como se puede ver de la playlist relacionado con el estado de ánimo ira o 'Anger' contamos con tan sólo 65 valores.  
                    Si vemos que más adelante nos da problemas el modelo entrenado eliminaremos o no tendremos en cuenta este estado de ánimo dependiendo del número de parámetros que se utilicen.  
                    En general, se suele decir que se necesitan al menos varias decenas o cientos de muestras de entrenamiento por cada variable de entrada (característica) que se utilice en el modelo. Esto se conoce como la regla de "diez veces el número de variables por muestra". Por ejemplo, si tienes 10 características, podrías necesitar al menos 100 muestras de entrenamiento.  
                    Así que, en principio, mi análisis se centrará en unos 10 parámetros aproximadamente para que se cumpla esta regla, por lo menos para los cuatro primeros estados de ánimo.  
                    https://postindustria.com/how-much-data-is-required-for-machine-learning/#:~:text=The%20most%20common%20way%20to,parameters%20in%20your%20data%20set.''')
        st.write('')
        st.write('')
        
        
        # En vez de usar fig.show() como estamos en streamlit utilizamos:
        st.plotly_chart(fig2, use_container_width=True)
        
    if page =='🔢 Variables numéricas':
        def main():
            st.title("Aplicación Power BI")
            powerbi_embed_code = """
            <iframe title="Report Section" width="1620" height="900" src="https://app.fabric.microsoft.com/view?r=eyJrIjoiNzZmZmFiNmEtN2E3ZS00OGY3LWJkOTEtYmU1OWI0NTEzMTZhIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9" frameborder="0" allowFullScreen="true"></iframe>    """
            st.markdown(powerbi_embed_code, unsafe_allow_html=True)

        if __name__ == "__main__":
            main()

        st.subheader('Correlaciones:')
        st.write('''La variable "valence", al ser la que marca supuestamente el estado de ánimo de las canciones, nos indicará aquellas variables relacionadas con los estados de ánimo.  
                 Es decir, aquellas variables que tengan correlación con "valence" tendrán relación con el estado de ánimo.''')     
        st.write('')
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plt)
        with col2:    
            st.write('')
        
        st.write('Según el gráfico podemos interpretar que:')

        st.markdown('''📈 Hay una :green[**fuerte correlación positiva**] con la variable "energy" y :green[**correlación positiva**] con las variables "popularity", "danceability" y "loudness".  
                    📉 Existe una :red[**correlación negativa**] con las variables "acousticness" e "instrumentalness"''')
           
    