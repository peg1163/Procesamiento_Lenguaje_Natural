## Objetivo del experimento

La idea del trabajo fue probar un escenario simple de preentrenamiento continuo con dos dominios de texto en español  
Usé un modelo base tipo distilgpt2 como modelo de lenguaje y lo fui afinando por etapas  

El esquema general fue este  

Dominio A  texto sobre IA  solo un párrafo  

Dominio B  texto más “especializado” sobre reglamento  también un párrafo  

Primero medí cómo se comportaba el modelo base en los dos dominios  
Luego lo entrené en A  Fase A  
Después continué el entrenamiento en B  Fase B  
En cada paso medí pérdida y perplejidad en A y en B para ver si aparecía algo parecido a catastrophic forgetting  


## Preparación de los datos y modelo

Para la data usé dos párrafos cortos  
uno sobre temas de IA y otro sobre reglamento  
La idea inicial era usar datasets grandes pero al intentar cargar datos más pesados empecé a tener problemas de espacio en disco y cache  
Por eso para esta primera entrega me quedé con la versión simple  

Aun así dejé armado un script `prepare_data.py` que en una versión ampliada genera  

- `data/dominio_a.txt` con texto general  
- `data/dominio_b.txt` con texto legal  

Pero en esta entrega solo usé los párrafos mencionados  

Luego con `data_utils.py` armé los conjuntos de entrenamiento y evaluación  
Ahí se hace la tokenización con el tokenizer de distilgpt2 se recorta a una longitud fija y se preparan los labels para lenguaje auto regresivo  

El modelo base se carga desde Hugging Face  
Todo el entrenamiento y evaluación se hace con `Trainer` de transformers para no pelearme con los loops a mano  


## Esquema de entrenamiento por fases

El flujo en `train.py` quedó así  

Primero evalúo el modelo base  

- Evalúo distilgpt2 tal cual en el test de A  esto es `base_A`  
- Evalúo lo mismo en el test de B  esto es `base_B`  

Luego viene la Fase A entrenamiento en dominio A  

- Tomo el modelo base y lo entreno solo con datos de A  
- Al final evalúo ese modelo en  
  - A  `after_A_A`  
  - B  `after_A_B`  

Después viene la Fase B entrenamiento continuo en dominio B  

- Empiezo desde el modelo ya entrenado en A  
- Lo sigo entrenando pero ahora solo con datos de B  
- Al final evalúo en  
  - A  `after_B_A`  
  - B  `after_B_B`  

En todas las evaluaciones uso la misma métrica  

- `eval_loss`  pérdida de lenguaje  
- `eval_perplexity`  perplejidad que es simplemente `exp(loss)`  

Perplejidad más baja significa que el modelo se equivoca menos al predecir el siguiente token  


## Resultados obtenidos

Estas fueron las métricas que guardé en `outputs_e1/metrics_e1.json`  

Modelo base  

- `base_A`  pérdida 5.1411  perplejidad 170.90  
- `base_B`  pérdida 5.0352  perplejidad 153.72  

Después de Fase A entrenado en A  

- `after_A_A`  pérdida 4.9909  perplejidad 147.07  
- `after_A_B`  pérdida 5.0020  perplejidad 148.71  

Después de Fase B entrenado en B empezando desde A  

- `after_B_A`  pérdida 4.9455  perplejidad 140.55  
- `after_B_B`  pérdida 4.9835  perplejidad 145.98  


Se puede leer así  

Con el modelo base  

- En A la perplejidad es 170.90  
- En B la perplejidad es 153.72  

O sea el modelo sin afinar se equivoca bastante en los dos párrafos  
y un poco menos en el párrafo de reglamento  

Después de la Fase A  

- En A baja de 170.90 a 147.07  
- En B baja de 153.72 a 148.71  

Aunque solo entrené con el párrafo de IA  
también hay una pequeña mejora cuando evalúo en el párrafo de reglamento  

Después de la Fase B  

- En A baja otra vez de 147.07 a 140.55  
- En B baja de 148.71 a 145.98  

Si comparo contra el modelo original  

- A pasa de 170.90 a 140.55  
- B pasa de 153.72 a 145.98  


## Interpretación y relación con catastrophic forgetting

La idea de catastrophic forgetting es que si entrenas el modelo primero en A y luego solo en B el modelo se especializa tanto en B que pierde rendimiento en A  

En este experimento no se ve ese efecto fuerte  
y tiene sentido porque los datos que usé son súper pequeños  básicamente dos párrafos  
No hay suficiente información como para que el modelo se especialice mucho en un dominio y destruya lo anterior  

Lo que se observa es  

- El rendimiento en A mejora después de Fase A y vuelve a mejorar un poco después de Fase B  
- El rendimiento en B también mejora después de Fase A y después de Fase B  

Al final el modelo ajustado en dos fases es mejor que el modelo base tanto en el texto de IA como en el texto de reglamento  
y también es mejor que el modelo justo después de Fase A  

Esto sugiere que en este setting el entrenamiento actúa más como un pequeño afinamiento general sobre estos dos textos  
No alcanza para ver un catastrophic forgetting claro  
pero sí sirve para comprobar que el pipeline de entrenamiento continuo A → B funciona y que las métricas se actualizan como se espera  

La idea es que esta primera entrega sirva como baseline simple  
y en la siguiente iteración probar con más datos y dominios más grandes para analizar mejor el problema de olvido catastrófico  
s agresivos y técnicas para mitigar el forgetting  
