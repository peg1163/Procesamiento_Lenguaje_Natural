# Reporte final

## 1. Introducción
Hice un mini-transformer, de tipo decoder-only, con un solo bloque, y probé dos formas de meter la posición: RoPE y sinusoidal
Como parte del proyecto 2, armé la atención multi-cabeza con máscara causal y un caché de K/V para acelerar la generación token por token. También medí latencia, para ver en números qué tan rápido va cada modo
La idea fue simple: entrenar en un corpus sintético, reproducible, comparar RoPE vs sinusoidal por calidad (perplexity), y mostrar cómo se comporta la atención cuando evaluamos todo de una sola vez, y cuando generamos de forma autoregresiva

---

## 2. Datos y tarea
La tarea es modelado de lenguaje carácter a carácter, o sea, adivinar el siguiente símbolo 
El corpus se genera de manera determinista, combinando dos tipos de ejemplos: patrones con `a b c ( )`, y expresiones como `A+B=` sin la respuesta, para obligar al modelo a seguir el hilo.  
Para reproducibilidad, guardo un SHA256 del corpus crudo más un SALT, en `out/CORPUS_SHA256.txt`, y con `verify-corpus` puedo regenerarlo y chequear que el hash coincida.  
Los splits son 90% train, 5% valid, 5% test.

---

## 3. Modelo (proyecto 3)
El bloque tiene: embeddings de tokens, una codificación posicional, la atención causal multi-cabeza, sumas residuales y normalización, un MLP con GELU, y al final, una proyección al vocabulario.

Sobre las posiciones:
- Sinusoidal, suma una señal fija que marca la posición, y listo.
- RoPE, en cambio, rota por parejas las dimensiones del embedding dependiendo de la posición, lo que ayuda a mantener la “forma” del vector y a captar relaciones relativas.

Para entrenar usé entropía cruzada, y reporto perplexity. Apliqué un pequeño warm-up al inicio, y recorte de gradiente, para evitar explosiones raras.

---

## 4. Atención y KV-cache (proyecto 2)
La atención calcula cuánto mira cada token a los anteriores, siempre con máscara causal para no “ver el futuro”.  
Cuando evaluamos todo junto, de punta a punta, es muy eficiente, porque el cómputo se vectoriza. Cuando generamos token por token, en cambio, conviene guardar en un caché los valores ya calculados, así no recomputamos todo el prefijo en cada paso. Ese caché, en la práctica, ahorra bastante trabajo a medida que crece el contexto.

---

## 5. Configuración de los experimentos
Para la ablación, corrí tres veces con RoPE y tres con sinusoidal, usando el mismo contexto, el mismo batch, y la misma tasa de aprendizaje.  
Para latencia, medí en CPU dos modos: “full” (un forward único de toda la secuencia) y “cache” (generación token a token con caché), para T=128 y T=256, con tres repeticiones por cada T.

---

## 6. Resultados

### 6.1 Perplexity en validación
| Posicional | Rep 1 | Rep 2 | Rep 3 | Promedio ± SD |
|---|---:|---:|---:|---:|
| RoPE | 9.226 | 9.343 | 9.211 | 9.260 ± 0.072 |
| Sinusoidal | 10.364 | 10.336 | 10.069 | 10.256 ± 0.163 |

RoPE queda alrededor de un punto por debajo de sinusoidal, en promedio, y además se ve más estable entre corridas.

### 6.2 Latencia del proyecto 2, en milisegundos
Promedios y desvío, con tres repeticiones por condición.

| T | full, un forward | cache, autoregresivo |
|---:|---:|---:|
| 128 | 1.85 ± 0.17 | 18.24 ± 0.50 |
| 256 | 4.27 ± 0.32 | 48.86 ± 5.12 |

El modo “full” es ideal para entrenamiento y validación, porque vectoriza todo y aprovecha mejor el hardware. El modo “cache”, aunque se ve más lento que el full, representa lo que pasa al generar texto de verdad, paso a paso, y evita recomputar el prefijo completo, que es donde se pierde tiempo.

---

## 7. Gráficos y artefactos
- Curva de pérdida, en `out/loss.png`.  
- Gráfico de latencia, en `out/plot_latencia.png`.  
Ambas imágenes se generan con `python run.py plot`

---

## 8. Análisis y discusión
RoPE, al trabajar con rotaciones dependientes de la posición, mantiene mejor la información relativa, y eso, en este dataset, se traduce en una perplexity más baja, y en menos variación entre corridas.  
Sobre la atención, el cálculo completo de una sola vez rinde muy bien, pero en generación, lo razonable es usar caché, porque cada token nuevo, solo necesita sumar
