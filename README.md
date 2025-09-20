# PC1_Proyecto7_Recuperacion_Embeddings


## TEORIA 
      *  Cosine similarity y normalización L2.
        la similitud del coseno mide el angulo entre vectores , esto permite comparar la direccion entre vetores  
        mientras que cuando normalizamos los vectores a norma l2 el producto punto ya nos datia la similitud del coseno 
      *  Define anisotropía en embeddings.
      cuando se generan embedings de un volumnen representaticvo de oraciones se espera que la dispersion de estas sea lo mas uniforme pero en la practica se tiene unaa concentracion de vectores en una region estrecha  o en un "cono " del espacio
      *  ¿Cuándo TF-IDF supera a embeddings?
        Estos puede ocurrir cuando se tiene un conjunto de datos reducido en donde hay presencia de palabras claves 
        y donde tiene un dominio especifico 
      *  Curse of dimensionality en recuperación.
        Cuando se tiene una dimensionalidad alta en el espacio de representacion de documentos , estas distancias entre las variables en tdf o embeddings tienden ser parecidas , haiendo que
        la diferencia entre vecinos sea mas cercano , o si hay algun lejano que su valor sea minimo haciendo asi que las metricas como el coseno de similituid  pierda el poder de discriminacion



      *  Impacto de stop-words en TF-IDF
        Son palabras muy comunes y que dentro de TF -IDF no aportan mucha informacion , se pueden eliminar y ayudaria a tener menos ruido pero de igual forma su peso asignado es pequeño