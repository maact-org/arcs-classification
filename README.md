# Requerimientos
Instalar requerimientos de python
```bash
pip install -r requirements
```
# Clasificador de sentimientos
Modelo clasificador por sentimientos
## Entrenar
### 1. Customización de la configuración del entrenamiento en archivo /settings.py
``` python
# ruta del data set usado para el entrenamiento, debe contener los campos mainDescription(str),code(str)
DATA_SET_PATH = 'hedonometer/data/sentiments_m_cleaned.csv'

#Longitud máxima para las secuencias de ID dadas por el tokenizador
MAX_LEN = 512

# Longitud del batch
BATCH_SIZE = 4

# Ruta del modelo a ser usado en el clasificador
PRE_TRAINED_MODEL = "bert-base-portuguese-cased/"

# Una cadena que especifica el dispositivo que se utilizará normalmente cuda: 0 o cpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Random
RANDOM_SEED = 123

#Ruta donde se guardará el modelo(incluye extensión)
SAVE_PATH = 'hedonometer/data/best_{}.bin'
```
### 2. Lectura de datos
``` python
 df = pd.read_csv(st.DATA_SET_PATH)
```
### 3. Asignar datos para entrenamiento y para evaluación del modelo
``` python
  df_raw_train, df_raw_test = train_test_split(
        df,
        test_size=0.1,
        random_state=st.RANDOM_SEED
    )
```
### 4. Creación del tokenizer
``` python
 tokenizer = BertTokenizer.from_pretrained(st.PRE_TRAINED_MODEL, do_lower_case=False)
```
### 5. Tokenizar datos
Datos de entrenamiento
``` python
  train_data_loader = get_prepared_dataset(df_raw_train, tokenizer, st.MAX_LEN, st.BATCH_SIZE)
    
```
Datos de evaluación
``` python
test_data_loader = get_prepared_dataset(df_raw_test, tokenizer, st.MAX_LEN, st.BATCH_SIZE)
```
### 6. Instanciar modelo
``` python
model = SentimentClassifier(st.PRE_TRAINED_MODEL, 'sentiment_m_cleaned')
model.to(st.DEVICE)
```    
### 7. Crear optimizador
``` python
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
```  
### 8. Crear función de pérdida
``` python
    loss_fn = nn.BCELoss().to(st.DEVICE)
``` 
### 9. Entrenar
``` python
    train_model(model, loss_fn, optimizer, train_data_loader, test_data_loader)
```  
## Predecir
### 1. Lectura de datos
El usuario puede realizar su propia función de lectura de datos dependiendo del tipo de almacenamiento que maneje para los datos
``` python
df = pd.read_csv('su_ruta/nombre.csv')
```
### 3. Creación del tokenizer
``` python
    tokenizer = BertTokenizer.from_pretrained(st.PRE_TRAINED_MODEL, do_lower_case=False)
```
### 4. Tokenizar datos
Cabe recordar que el tokenizer recibe datos de tipo string
``` python
data_loader = get_prepared_dataset(df, tokenizer, st.MAX_LEN, st.BATCH_SIZE)
```
### 5. Instanciar modelo
``` python
model = SentimentClassifier(st.PRE_TRAINED_MODEL, 'sentiment_m_cleaned')
model.to(st.DEVICE)
```    
### 6. Cargar diccionario de estados
``` python
model.load_state_dict(torch.load('su_ruta/modelo.bin', map_location='cpu'))
```  
### 7. Predecir
Se retorna un array de 1 posiciones por cada libro que contiene una probabilidad. Si es mayor a 5 el sentimiento es positivo, de lo contrario es negativo.
``` python
   predictions = []
   preds = []
   with torch.no_grad():
      for d in data_loader:
         input_ids = d["input_ids"].to(st.DEVICE)
         attention_mask = d["attention_mask"].to(st.DEVICE)
         outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask
         )
         predictions += [outputs]
         for pre in predictions:
               preds += [pre.cpu().numpy()]
         preds = np.asarray(preds)
         data = np.vstack((pred for pred in preds))
``` 
### 8. Disponer datos de salida como se requiera
El usuario debe escoger la forma de almacenamiento para los datos de salida que más se acomode a sus necesidades

# Clasificador de arcos
Modelo clasificador de textos por arcos teniendo en cuenta su clasificación por sentimientos

## Entrenar
### 1. Customización de la configuración del entrenamiento en archivo /settings.py
``` python
# ruta del data set usado para la evaluación del modelo
DATA_SET_EVAL_PATH = 'data/Arcos'

#Longitud máxima para las secuencias de ID dadas por el tokenizador
MAX_LEN = 512

# Longitud del batch
BATCH_SIZE = 1

# Ruta del modelo a ser usado en el clasificador
BERT_MODEL_PATH  = "bert-base-portuguese-cased"

# Ruta del modelo almacena .bin
SC_MODEL_PATH = "models/best_sentiment_m_cleaned.bin"

#Ruta del data loader tokenizado
DATALOADER_TOKENIZED_PATH = 'tokenized.torch'
```

### 2. Instanciar modelo
``` python
model = ArcModel(input_shape=(33, 1))
```    
### 3. Entrenar
``` python
model.model = trn.train_model_with_csv(
               model.model,
               file_path="training/arcs.csv",
               epochs=500)
```  
### 4. guardar
``` python
model.save_model(file='data/arcs_classifier.h5')
``` 
## Predecir
El ejemplo se encuentra en arcs-classification/example_analyzer.py
### 1. Lectura de datos
``` python
file_names = os.listdir(EXAMPLE_PATH)
files = [open(EXAMPLE_PATH + name) for name in file_names]
books = [file.read() for file in files]
```
### 2. Instanciar modelo
``` python
model = ArcModel(input_shape=(33, 1))
```    
### 3. Cargar diccionario de estados
``` python
model.load_model(file='data/arcs_classifier.h5')
```  
### 4. Tokenizar y pasar por el modelo de sentimientos
``` python
    books = [file.read() for file in files]
    anlzr = Analyzer(
        hedonometer_path=HEDONOMETER,
        tokenizer_path=st.PRE_TRAINED_MODEL
    )
```
### 5. Predecir
Sólo un libro
``` python
 arc = anlzr.predict_book_arc(books[0])
``` 
Varios libros
``` python
 arcs = anlzr.predict_multiple_books(books)
``` 
Se retorna un array de 6 posiciones por cada libro que contiene la probabilidad del libro por cada arco.
### 6. Disponer datos de salida como se requiera
El usuario debe escoger la forma de almacenamiento para los datos de salida que más se acomode a sus necesidades
