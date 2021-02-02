# People Counting Using Visible And Infrared Images

***
- [People Counting Using Visible And Infrared Images](#people-counting-using-visible-and-infrared-images)
  - [Descripción](#descripción)
  - [Requisitos](#requisitos)
    - [Dataset Training](#dataset-training)
    - [Dataset Validation](#dataset-validation)
    - [Dataset Testing (opcional)](#dataset-testing-opcional)
    - [Dataset de imágenes generadas digitalmente (opcional)](#dataset-de-imágenes-generadas-digitalmente-opcional)
    - [Características técnicas](#características-técnicas)
    - [Estructura de directorios](#estructura-de-directorios)
    - [Formato de archivos por directorio](#formato-de-archivos-por-directorio)
  - [Manual de uso del código fuente](#manual-de-uso-del-código-fuente)
    - [images_blurring.py](#images_blurringpy)
      - [Parámetros](#parámetros)
      - [Output](#output)
      - [Ejemplo de uso](#ejemplo-de-uso)
    - [positions_extractor.py](#positions_extractorpy)
      - [Parámetros](#parámetros-1)
      - [Output](#output-1)
      - [Ejemplo de uso](#ejemplo-de-uso-1)
    - [labels_generator.py](#labels_generatorpy)
      - [Parámetros](#parámetros-2)
      - [Output](#output-2)
      - [Ejemplo de uso](#ejemplo-de-uso-2)
    - [data_paths_generator.py](#data_paths_generatorpy)
      - [Parámetros](#parámetros-3)
      - [Output](#output-3)
      - [Ejemplo de uso](#ejemplo-de-uso-3)
    - [training.<span>py</span>](#trainingpy)
      - [Parámetros](#parámetros-4)
      - [Output](#output-4)
      - [Ejemplo de uso](#ejemplo-de-uso-4)
    - [testing.<span>py</span>](#testingpy)
      - [Parámetros](#parámetros-5)
      - [Output](#output-5)
      - [Ejemplo de uso](#ejemplo-de-uso-5)

***
## Descripción

Este proyecto consiste en un software para el conteo y ubicación de personas en imágenes compuestas por los canales visibles (RGB) y el canal infrarrojo (IR). Para esto, se utiliza una Red Neuronal Convolucional (CNN) de arquitectura perteneciente al grupo de las U-Net.
Se cuenta con un conjunto de *scripts* en lenguaje Python para ejecutar los diferentes pasos que se deben seguir para entrenar y, posteriormente, utilizar la red.

***
## Requisitos

### Dataset Training

Para poder entrenar la red, se necesita contar con los siguientes elementos:

1. Set de imágenes con información de los canales visibles (RGB).
2. Set de imágenes con información del canal infrarrojo (IR) donde cada imagen tenga su par dentro del set de imágenes RGB (con las personas en las mismas posiciones).
3. Set de archivos *.json* con las posiciones de las personas.

### Dataset Validation

Para poder verificar qué configuración de pesos de la red durante su entrenamiento es mejor que otra, se utiliza un dataset independiente (IDEM 3 elementos del dataset de *Training*).

### Dataset Testing (opcional)

Una vez que la red se encuentra entrenada se puede analizar una única imagen, obteniendo las posiciones de todas las personas que reconoce la red. Además, se puede analizar un set de imágenes con el propósito de calcular el error entre las predicciones de la red y la realidad. En este último escenario, se necesita un tercer dataset (IDEM 3 elementos del dataset de *Training*).

### Dataset de imágenes generadas digitalmente (opcional)

Debido a la poca cantidad de imágenes reales de multitudes de personas (y que cuenten con información de los 4 canales RGB+IR que se necesitan) con las que contamos, se utilizaron imágenes generadas digitalmente a partir de imágenes reales. En consecuencia, se incluye un *script* que le agrega ruido y distorción a estas imágenes para evitar que la red reconozca bordes o patrones causados por el software generador de imágenes.

### Características técnicas

Para poder ejecutar el código que entrena la red neuronal, es necesario contar con una GPU compatible con torch CUDA https://pytorch.org/docs/stable/cuda.html.
En nuestro caso, utilizamos el equipo del itba *titan.it.itba.edu.ar* que cuenta con una placa TITAN.

### Estructura de directorios

Para el entrenamiento de la red neuronal y su utilización para testearla con un conjunto de imágenes, utilizar la siguiente estructura de directorios:

**code**: contiene todos los scripts python del código fuente.  
\- \- **logs**: contiene los logs de las distitnas ejecuciones.  
**data**: contiene todos los archivos involucrados en el entrenamiento (imágenes, posiciones de personas, etc).  
\- \- **training**: archivos relacionados con el entrenamiento.  
\- \- \- \- **images**  
\- \- \- \- \- \- **raw**: imágenes sin ruido.  
\- \- \- \- \- \- \- \- **rgb**: canales visibles.  
\- \- \- \- \- \- \- \- **ir**: canal infrarojo.  
\- \- \- \- \- \- **processed**: imágenes luego de aplicar ruido.  
\- \- \- \- \- \- \- \- **rgb**: canales visibles.  
\- \- \- \- \- \- \- \- **ir**: canal infrarojo.  
\- \- \- \- **labels**: *ground-truths* que utilizará la red para comparar con su *output*.  
\- \- \- \- **particles**: archivos .json con información de las personas (tratadas como partículas) de las imágenes.  
\- \- \- \- **positions**: archivos .h5 generados al extraer las posiciones de las personas de la carpeta **particles**.  
\- \- **testing**: archivos relacionados con el testeo de la red.  
\- \- \- \- **images**: IDEM **training**.  
\- \- \- \- **labels**: IDEM **training**.  
\- \- \- \- **particles**: IDEM **training**.  
\- \- \- \- **positions**: IDEM **training**.  
\- \- \- \- **predictions**: archivos .h5 con las predicciones hechas por la red (cantidad de personas y sus posiciones).  
\- \- \- \- \- \- **rgb**: contendrá predicciones hechas por red entrenada con 3 canales.  
\- \- \- \- \- \- **rgb-ir**: contendrá predicciones hechas por red entrenada con 4 canales.  

### Formato de archivos por directorio

**images.raw.rgb**: rgb\_image\_{id}\_{altura}.jpg, donde **id** es un número de 7 dígitos y **altura** es una letra que identifica la altura de la imágen.  
**images.raw.<span>ir</span>**: ir\_image\_{id}\_{altura}.jpg.  
**particles**: particles\_{id}\_{altura}.json.  


***
## Manual de uso del código fuente

La ejecución del código fuente está separada en varios <b>scripts</b>, los cuales se deben ejecutar en un cierto orden dependiendo lo que se quiera realizar. A continuación se listan los <b>scripts</b>. Luego, se detallan cuáles y en qué orden deben ejecutarse tanto para entrenar la red neuronal como para utilizarla.

### images_blurring.py

Permite aplicar ruido y *blurring* a un conjunto de imágenes RGB y a otro conjunto IR. Para el *blurring* utiliza la función GaussianBlur encontrada en https://docs.opencv.org/master/d4/d86/group\_\_imgproc__filter.html. Luego, el ruido se logra con un valor aleatorio por píxel en el rango [Ip ∗ (1 − β), Ip ∗ (1 + β)] donde Ip es el valor del píxel y β es el parámetro --noise_eta.

#### Parámetros

| Parámetro | Tipo de dato | Obligatorio | Descripción |
|:--------- |:----------------------:|:-----------:|:-----------:|
| \-\-kernel_size | string | si | ksize utilizado por la función GaussianBlur |
| \-\-sigma | integer | si | sigma utilizado por la función GaussianBlur |
| \-\-noise_eta | float | si | explicado en la descripción del script |
| \-\-raw_images_dir | string | si | directorio con las imágenes sin procesar (debe contener los directorios /rgb y /ir con las imágenes jpg correspondientes) |
| \-\-processed_images_dir | string | si | directorio donde se dejarán las imágenes procesadas (debe contener los directorios /rgb y /ir) |

#### Output

Quedaran imágenes jpg con distorción y ruido dentro de los directorios correspondientes de la carpeta indicada por el parámetro **--processed_images_dir**.

#### Ejemplo de uso

```bash
nohup python3 -u images_blurring.py --kernel_size 7 --sigma 1 --noise_eta 0.1 --raw_images_dir ../data/training/images/raw --processed_images_dir ../data/training/images/processed > logs/training/images_bluring.log &
```

---
### positions_extractor.py

Extrae las coordenadas x e y de todas las personas de las imágenes. Estas posiciones las extrae de los archivos .json del directorio indicado.

#### Parámetros

| Parámetro | Tipo de dato | Obligatorio | Descripción |
|:--------- |:----------------------:|:-----------:|:-----------:|
| \-\-particles_dir | string | si | contiene los archivos .json |
| \-\-positions_dir | string | si | en donde se depositarán los archivos .h5 con las coordenadas de cada persona por imagen |

#### Output

Quedarán las coordenadas de cada persona de cada persona de cada imagen en archivos .h5

#### Ejemplo de uso

```bash
nohup python3 -u positions_extractor.py --particles_dir ../data/training/particles --positions_dir ../data/training/positions > logs/training/positions_extractor.log &
```

---
### labels_generator.py

Genera los archivos *ground-truth* .h5 que se va a usar la red para comparar contra sus *outputs* durante el entrenamiento.

#### Parámetros

| Parámetro | Tipo de dato | Obligatorio | Descripción |
|:--------- |:----------------------:|:-----------:|:-----------:|
| \-\-positions_dir | string | si | en donde se encuentran los archivos .h5 con las coordenadas de cada persona por imagen |
| \-\-images_dir | string | si | directorio con las imágenes procesadas (se necesita para hallar las dimensiones de las imágenes) |
| \-\-labels_dir | string | si | en donde se depositaran los archivos .h5 *ground-truth* |

#### Output

Quedarán los archivos *ground-truth* en la carpeta indicada por el parámetro **--labels_dir** con extensión .h5.

#### Ejemplo de uso

```bash
nohup python3 -u labels_generator.py --positions_dir ../data/training/positions --images_dir ../data/training/images/processed --labels_dir ../data/training/labels > logs/training/labels_generator.log &
```

---
### data_paths_generator.py

Genera archivos con los paths a todas las imágenes que se usarán para entrenar y validar la red.

#### Parámetros

| Parámetro | Tipo de dato | Obligatorio | Descripción |
|:--------- |:----------------------:|:-----------:|:-----------:|
| \-\-images_dir | string | si | directorio con las imágenes procesadas |
| \-\-training_file | string | si | archivo donde se guardarán los paths a las imágenes de entrenamiento |
| \-\-validating_file | string | si | archivo donde se guardarán los paths a las imágenes de validación |
| \-\-validating_count | integer | si | cantidad de imágenes del total que se destinarán a validación |

#### Output

1.  Archivo con una lista de paths a las imágenes necesarias para el entrenamiento de la red neuronal.
2.  Archivo con una lista de paths a las imágenes necesarias para la etapa de validación del entrenamiento.

#### Ejemplo de uso

```bash
nohup python3 -u data_paths_generator.py --images_dir ../data/training/images/processed --training_file 'training_paths.json' --validating_file 'validating_paths.json' --validating_count 130 > logs/training/data_paths_generator.log &
```

---
### training.<span>py</span>

Entrena una red neuronal utilizando los sets de imágenes output del script **data_paths_generator** correspondientes a los conjuntos de entrenamiento y validación. Se puede entrenar la red con 3 (RGB) o 4 (RGB + IR) canales.

#### Parámetros

| Parámetro | Tipo de dato | Obligatorio | Default | Descripción |
|:--------- |:----------------------:|:-----------:|:-------:|:-----------:|
| \-\-learning_rate | float | no | 0.02 | *learning rate* utilizado en el entrenamiento |
| \-\-batch_size | integer | no | 1 | cantidad de muestras que se procesan antes de actualizar los pesos de la red |
| \-\-momentum | float | no | 0.9 | *momentum* utilizado en el entrenamiento |
| \-\-nesterov | boolean | no | False | indica si se utiliza la estrategia de Nesterov Momentum |
| \-\-weight_decay | float | no | 0 | *weight decay* utilizado en el entrenamiento |
| \-\-step_size | integer | no | 10 | indica la cantidad de *epochs* que deberán transcurrir para actualizar el *learning rate* |
| \-\-gamma | float | no | 0.1 | cada *step_size epochs* se actualiza el *learning rate* siendo $lr = lr * gamma$ |
| \-\-epoch_start | integer | no | 0 | indica el número de *epoch* del entrenamiento (utilizado al reanudar entrenamiento desde archivo de *checkpoint*) |
| \-\-epoch_end | integer | no | 50 | indica la cantidad de *epochs* que deberán transcurrir hasta que se frena la ejecución (se puede frenar de forma manual eliminando el proceso) |
| \-\-print_freq | string | no | 50 | cada *print_freq* imágenes procesadas se imprime un log con información del entrenamiento |
| \-\-ir_enabled | boolean | no | False | indica si se debe entrenar la red con 3 (RGB) o 4 (RGB + IR) canales |
| \-\-start_from_checkpoint | string | no | False | indica si se debe reanudar un entrenamiento a partir de un archivo de *checkpoint* (que se debe proporcionar) |
| \-\-training_file | string | si | - | archivo con los paths a las imágenes utilizadas en el entrenamiento |
| \-\-validating_file | string | si | - | archivo con los paths a las imágenes utilizadas en la validación |
| \-\-checkpoint_filename | string | si | - | archivo de *checkpoint* de la configuración de pesos de la red |
| \-\-mse_history_filename | string | si | - | archivo en el que se guardará la evolución del *mean square error* calculado en la etapa de validación |
| \-\-images_dir | string | si | - | directorio con las imágenes procesadas |
| \-\-labels_dir | string | si | - | directorio con los archivos .h5 que contienen los *ground-truths* de las imágenes |

#### Output

1.  Un archivo con formato de *filename* **{--checkpoint_filename}** con la última configuración de pesos de la red (de la última *epoch* ejecutada).
2.  Un archivo con formato de *filename* **best_{--checkpoint_filename}** con la **mejor** configuración de pesos hasta el momento (la de menor *mean square error*).

#### Ejemplo de uso

```bash
nohup python3 -u training.py --training_file 'training_paths.json' --validating_file 'validating_paths.json' --momentum 0.9 --nesterov True --checkpoint_filename 'rgb_weights.tar' > logs/training/training-rgb.log &
```

---
### testing.<span>py</span>

Permite testear la red procesando un dataset de *testing* y extrayendo la cantidad de personas predichas y las posiciones de todas las personas de cada imagen.

#### Parámetros

| Parámetro | Tipo de dato | Obligatorio | Default | Descripción |
|:--------- |:----------------------:|:-----------:|:-------:|:-----------:|
| \-\-trained_network | string | si | - | archivo con la configuración de pesos de la red entrenada |
| \-\-threshold | float | no | 30 | valor a partir de cual se considerará que un píxel (sumado a sus vecinos) representan a una persona |
| \-\-ir_enabled | boolean | no | False | indica si se va a testear con 3 o 4 canales (debe coincidir con la configuración de la red entrenada) |
| \-\-images_dir | string | si | - | directorio con las imágenes procesadas |
| \-\-labels_dir | string | si | - | directorio con los archivos .h5 *ground-truth* |
| \-\-positions_dir | string | si | - | directorio con los archivos .h5 con las posiciones reales de todas las personas por imagen |
| \-\-predictions_dir | string | si | - | directorio donde se generarán las predicciones de la red sobre el dataset de *testing* |

#### Output

Archivos .h5 con las predicciones hechas por la red neuronal luego de procesar el dataset de *testing*. Estas predicciones incluyen cantidad de personas y la posición de todas las personas por imagen.

#### Ejemplo de uso

```bash
nohup python3 -u testing.py --trained_network './best_rgb_weights.tar' --threshold 30 --images_dir ../data/testing/images/processed --labels_dir ../data/testing/labels --positions_dir ../data/testing/positions --predictions_dir ../data/testing/predictions > logs/testing/testing_rgb.log &
```

