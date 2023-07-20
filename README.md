![Logo Image](./img/logo.png)
## Introducción:  
Este proyecto pretende analizar una serie de playlists de Spotify a través de su API y la librería de Python spotipy.  


Las playlists han sido seleccionadas con intención de analizar sus características según el estado de ánimo que representan para así poder entrenar un modelo que recomiende una canción según tu estado de ánimo.  

## Características de las canciones:  
| audio_feature | Descripción | Valores |
|---|---|---|
| **Acousticness** | 1,0 representa una confianza alta en que la pista es acústica. | 0,0 a 1,0 |
| **Danceability** | Un valor de 0,0 es el menos bailable y 1,0 el más bailable. | 0,0 a 1,0 |
| **duration_ms** | Duración de la pista en milisegundos. | - |
| **Energy** | Cuanto más se acerque el valor a 1,0 más enérgica es la canción. | 0,0 a 1,0 |
| **Instrumentalness** | Cuanto más se acerque el valor a 1,0 mayor será la probabilidad de que la pista no contenga voces. | 0,0 a 1,0 |
| **Key** | La tonalidad de la pista. | 0 = C, 1 = C♯/D♭, 2 = D, y así sucesivamente. Ninguna clave = -1 |
| **Liveness** | > 0,8 proporciona una gran probabilidad de que la pista sea en directo. | 0,0 a 1,0 |
| **Loudness** | La sonoridad global de una pista en decibelios (dB). | -60 a 0 db |
| **Mode** | Mayor se representa con 1 y menor con 0. | 1 o 0 |
| **Speechiness** | <0,33 canciones no habladas / 0,33 - 0,66 música y voz / >0,66 canciones habladas | 0,0 a 1,0 |
| **Tempo** | Pulsaciones por minuto (BPM) | - |
| **Valence** | Cuanto más alto más alegre es la pista.  |0,0 a 1,0|
| **time_signature** | Compáses de "3/4" a "7/4" | 3 y 7 |
| **Popularity** | Siendo 100 la canción más popular. | 0 a 100 |
 
**Danceability**: La bailabilidad describe lo adecuada que es una pista para bailar basándose en una combinación de elementos musicales como el tempo, la estabilidad del ritmo, la fuerza del compás y la regularidad general. Un valor de 0,0 es el menos bailable y 1,0 el más bailable.  
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

## Bibliografía:  
Librería spotipy: https://spotipy.readthedocs.io/en/2.19.0/  
Características de las pistas: https://developer.spotify.com/documentation/web-api/reference/get-audio-features  
Gifs: https://medium.com/@nuriaalcarazesteve/spotify-nueva-funcionalidad-8ad74cbad77e
Powerbi:  
Logos: https://developer.spotify.com/documentation/design  
Iconos: https://www.freepik.com/icon/group_1464029#position=95&page=4&term=mood&fromView=keyword  
Fondo lienzo: https://www.pexels.com/photo/grayscale-piano-keys-159420/  
Akkio: https://app.akkio.com/  
Fuentes útiles: https://rstudio-pubs-static.s3.amazonaws.com/716221_92850e1ae9224bb0b6f0e2a58b42f9b4.html  
https://towardsdatascience.com/extracting-song-data-from-the-spotify-api-using-python-b1e79388d50  
https://postindustria.com/how-much-data-is-required-for-machine-learning/#:~:text=The%20most%20common%20way%20to,parameters%20in%20your%20data%20set  
PCA: https://scikit-learn.org/stable/modules/decomposition.html#pca  
Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest
