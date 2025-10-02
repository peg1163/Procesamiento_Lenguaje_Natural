import pandas as pd
import random
import numpy as np

# Fijar semilla para reproducibilidad
random.seed(42)
np.random.seed(42)

# Listas de palabras para generar oraciones
temas_nlp = ['tokenización', 'embeddings', 'modelos de lenguaje', 'LLMs', 'transformers', 'clasificación', 'regularización', 'perplejidad', 'BPE', 'lematización']
adjetivos_positivos = ['fascinante', 'útil', 'innovador', 'impresionante', 'eficiente', 'claro', 'esencial']
adjetivos_negativos = ['complicado', 'confuso', 'difícil', 'lento', 'limitado', 'frustrante']
adjetivos_neutrales = ['interesante', 'técnico', 'complejo', 'fundamental', 'necesario']
verbos = ['es', 'parece', 'resulta', 'se usa para', 'ayuda a', 'requiere', 'mejora']

# Plantillas para oraciones
plantillas = [
    "La {tema} {verbo} {adjetivo} para procesar texto.",
    "Entender los {tema} {verbo} {adjetivo} en el curso de NLP.",
    "Los {tema} son {adjetivo} pero {adjetivo_complementario}.",
    "Implementar {tema} {verbo} {adjetivo} en proyectos reales.",
    "No entiendo cómo funciona la {tema}, es {adjetivo}.",
]

# Generar oraciones
def generar_oracion(categoria):
    plantilla = random.choice(plantillas)
    tema = random.choice(temas_nlp)
    verbo = random.choice(verbos)
    if categoria == 'Positivo':
        adjetivo = random.choice(adjetivos_positivos)
        adjetivo_complementario = random.choice(adjetivos_positivos + adjetivos_neutrales)
    elif categoria == 'Negativo':
        adjetivo = random.choice(adjetivos_negativos)
        adjetivo_complementario = random.choice(adjetivos_negativos + adjetivos_neutrales)
    else:  # Neutral
        adjetivo = random.choice(adjetivos_neutrales)
        adjetivo_complementario = random.choice(adjetivos_neutrales + adjetivos_positivos)
    return plantilla.format(tema=tema, verbo=verbo, adjetivo=adjetivo, adjetivo_complementario=adjetivo_complementario)

# Generar dataset
n_oraciones = 10000
categorias = ['Positivo', 'Negativo', 'Neutral']
proporciones = [0.4, 0.3, 0.3]  # Distribución aproximada
oraciones = []
labels = np.random.choice(categorias, size=n_oraciones, p=proporciones)

for label in labels:
    oraciones.append(generar_oracion(label))

# Crear DataFrame
df = pd.DataFrame({'Texto': oraciones, 'Categoría': labels})

# Añadir ejemplos reales (inspirados en el dataset original)
ejemplos_reales = [
    ("La tokenización es clave para procesar texto", "Positivo"),
    ("No entiendo los embeddings vectoriales", "Negativo"),
    ("Los LLMs son impresionantes pero complejos", "Neutral"),
    ("El curso de NLP es fascinante y útil", "Positivo"),
    ("La programación en Python es complicada al principio", "Negativo"),
]
df = pd.concat([pd.DataFrame(ejemplos_reales, columns=['Texto', 'Categoría']), df]).reset_index(drop=True)

# Guardar dataset
df.to_csv('nlp_prueba_cc0c2_large.csv', index=False)
print(f"Dataset generado: {len(df)} oraciones en 'nlp_prueba_cc0c2_large.csv'")