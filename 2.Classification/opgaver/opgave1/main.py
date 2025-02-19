# Dette script downloader et billede-datasæt, forbereder trænings- og validationsdata,
# træner en simpel convolutional neural netværksmodel, visualiserer træningsforløbet,
# og tester modellen på et eksternt billede.

import matplotlib.pyplot as plt    # Matplotlib: bruges til at plotte grafer og billeder.
import numpy as np                 # NumPy: bruges til numeriske beregninger og håndtering af arrays.
import PIL                         # Pillow (PIL): bruges til billedbehandling.
import tensorflow as tf            # TensorFlow: et bibliotek til maskinlæring.
from tensorflow import keras       # Keras API'et gør det nemt at bygge og træne neurale netværk.
from tensorflow.keras import layers  # Importerer lag som fx convolutional lag til vores model.
from tensorflow.keras.models import Sequential  # Sequential-modellen gør det muligt at bygge modellen lag-for-lag.
import os                          # OS-modulet: til interaktion med operativsystemet og filstier.
import pathlib                     # pathlib: til håndtering af filstier på en platform-uafhængig måde.
import shutil                      # shutil: til funktioner for høj-niveau filhåndtering, fx sletning af mapper.
from util.display_image import display_images  # Hjælpefunktion til eventuel visning af billeder.
from util.visulizer import visualize_dataset     # Hjælpefunktion til visualisering af et datasæt.

# ------------------------------------------------------------
# Opsætning af datasæt-stien og forberedelse af arbejdsmappe
# ------------------------------------------------------------

# Få stien til den mappe, hvor scriptet ligger.
base_dir = os.path.dirname(os.path.abspath(__file__))
# os.path.abspath(__file__) returnerer den absolutte sti til denne fil.
# os.path.dirname() udtrækker mappenavnet fra den fulde sti.

# Kombinér base_dir med navnet 'datasets' til at danne datasætmappen.
dataset_dir = os.path.join(base_dir, 'datasets')
# os.path.join() sammensætter stier korrekt afhængig af operativsystemet (f.eks. '\' på Windows).

# Hvis mappen 'datasets' findes, slettes den for at starte med et rent datasæt.
if os.path.exists(dataset_dir):
    # os.path.exists() tjekker, om mappen findes.
    shutil.rmtree(dataset_dir)  
    # shutil.rmtree() sletter mappen rekursivt, dvs. alle undermapper og filer fjernes.
    print("Cleaned up existing dataset directory")

# Opret en ny 'datasets' mappe.
os.makedirs(dataset_dir, exist_ok=True)
# os.makedirs() opretter mappen. Parameteren 'exist_ok=True' betyder, at der ikke opstår en fejl,
# hvis mappen allerede eksisterer.

# ------------------------------------------------------------
# Download og udpakning af datasæt
# ------------------------------------------------------------

# URL til TGZ-arkivet med blomsterbilleder fra TensorFlow.
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# Download og udpak datasættet. Funktionen tf.keras.utils.get_file() downloader filen og udpakker den (hvis untar=True).
data_dir = tf.keras.utils.get_file(
    origin=dataset_url,        # URL'en, hvorfra filen downloades.
    fname='flower_photos',     # Navnet på den lokale fil eller mappe, hvor data gemmes.
    untar=True,                # Hvis True pakkes arkivet automatisk ud.
)

# Konverter den returnerede sti til et pathlib objekt, som gør stihåndtering nemmere.
data_dir = pathlib.Path(data_dir).resolve()
# pathlib.Path() opretter et objekt til stien, og .resolve() sikrer, at stien er absolut.

# Udskriv datasættets placering for at bekræfte, at download og udpakning lykkedes.
print(f"Using data directory: {data_dir}")

# Hvis datasættets mappe ikke findes, stoppes programmet med en fejl.
if not data_dir.exists():
    raise RuntimeError(f"Data directory not found: {data_dir}")
    # raise udløser en undtagelse (exception), som stopper programmet.

# ------------------------------------------------------------
# Tæl antallet af billeder i datasættet
# ------------------------------------------------------------
image_count = len(list(data_dir.rglob('*.jpg')))
# data_dir.rglob('*.jpg') søger rekursivt efter alle filer med ending .jpg.
# list() konverterer generatoren til en liste, og len() returnerer antallet af billeder.
print(f"Total images found: {image_count}")

# ------------------------------------------------------------
# Opsætning af datasættets parametre (batch størrelse og billedstørrelse)
# ------------------------------------------------------------
batch_size = 32        # Batch-størrelse: antal billeder per batch under træning.
img_height = 180       # Højde på billeder, som de skaleres til.
img_width = 180        # Bredde på billeder, som de skaleres til.

# ------------------------------------------------------------
# Oprettelse af trænings- og validationsdatasæt
# ------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),           # Datasættets placering.
    validation_split=0.2,      # 20% af data bruges til validation, 80% til træning.
    subset="training",         # Angiv, at dette er træningsdatasættet.
    seed=123,                  # Seed sikrer, at data deles konsekvent ved hver kørsel.
    image_size=(img_height, img_width),  # Alle billeder skaleres til størrelse (180x180).
    batch_size=batch_size      # Antal billeder pr. batch.
)
# tf.keras.utils.image_dataset_from_directory():
# Denne funktion læser billeder fra mapper, hvor hver undermappe repræsenterer en klasse.

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),
    validation_split=0.2,
    subset="validation",       # Angiv, at dette er validationsdatasættet.
    seed=123,                  # Samme seed sikrer, at data-split er identisk.
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# ------------------------------------------------------------
# Få listen over klassenavne (mappenavne fra datasættet)
# ------------------------------------------------------------
class_names = train_ds.class_names
print("Available classes:", class_names)

# Udskriv dataset-information: antal batches i trænings- og validationsdatasættet.
print("\nDataset information:")
print(f"Number of training batches: {len(train_ds)}")
print(f"Number of validation batches: {len(val_ds)}")

# ------------------------------------------------------------
# Visualisering af træningsdatasættet
# ------------------------------------------------------------
# Kald af hjælpefunktion fra visulizer.py der viser et 3x3 grid med billeder.
visualize_dataset(train_ds, class_names, num_images=9, figsize=(10, 10))
# visualize_dataset() modtager følgende:
#  - dataset: træningsdatasættet.
#  - class_names: navne på klasserne.
#  - num_images: antal billeder, der skal vises (her 9 ⇒ 3x3 grid).
#  - figsize: figurens størrelse.

# ------------------------------------------------------------
# Forbedring af datasættets ydelse
# ------------------------------------------------------------
# Brug af cache, shuffle og prefetch optimerer dataindlæsning under træning.
AUTOTUNE = tf.data.AUTOTUNE   # Automatisk tuning af bufferstørrelse.
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# cache() gemmer resultatet af dataindlæsning, så det ikke skal læses fra disk hver gang.
# shuffle(1000) blander data med en buffer på 1000 elementer for at undgå forudsigelige mønstre.
# prefetch() forbereder næste batch, mens den nuværende batch behandles.
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ------------------------------------------------------------
# Opret normaliseringslag til at skalere pixelværdier fra [0,255] til [0,1]
# ------------------------------------------------------------
normalization_layer = layers.Rescaling(1./255)
# layers.Rescaling() er et forbehandlingslag, som dividerer pixelværdier (0-255) med 255,
# hvilket resulterer i værdier mellem 0 og 1. Dette hjælper med en bedre træningsstabilitet.

# Anvend normaliseringen på træningsdatasættet med map()-funktionen.
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# map() anvender en funktion (her en lambda, som er en anonym funktion)
# på hvert element i datasættet, dvs. normaliserer hvert billede.

# Test normaliseringen ved at hente det første batch og udskrive min./max. pixelværdi.
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))
# next(iter(...)) henter det første element fra den iterator, dvs. det første batch.

# ------------------------------------------------------------
# Definer modelarkitektur
# ------------------------------------------------------------
# Antal klasser bestemmes ved længden af class_names-listen.
num_classes = len(class_names)

# Byg en simpel Convolutional Neural Network (CNN) model med Sequential API.
model = Sequential([
  # Første lag: Normaliserer input-billederne.
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  
  # Andet lag: Convolutional lag.
  # Et convolutional lag anvender filtre til at udtrække features fra billeder.
  # Filtre (eller kerner) er små matricer (fx 3x3) der "scanner" billedet og fremhæver mønstre.
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  # '16' betyder, at der bruges 16 filtre, hvilket resulterer i 16 feature-maps.
  # 'padding="same"' bevarer input-dimensionerne.
  # 'relu'-aktiveringsfunktionen introducerer ikke-linearitet og hjælper med at fange komplekse mønstre.
  
  # MaxPooling2D: Reducerer dimensionsstørrelsen ved at tage maksimumsværdien i et bestemt område.
  layers.MaxPooling2D(),
  
  # Tredje lag: Et andet convolutional lag med 32 filtre.
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Fjerde lag: Et convolutional lag med 64 filtre.
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Flatten: Omformer 2D-feature maps til en 1D-vektor.
  # Dette er nødvendigt for at forbinde til de fuldt forbundne (dense) lag.
  layers.Flatten(),
  
  # Dense lag: Et fuldt forbundet lag med 128 neuroner.
  # Dense-laget træffer beslutninger baseret på de features, der er udtrukket af de convolutionale lag.
  layers.Dense(128, activation='relu'),
  
  # Output-lag: Et lag med et neuron for hver klasse.
  # Outputtet kaldes 'logits', der ikke er normaliserede sandsynligheder.
  layers.Dense(num_classes)
])
  
# ------------------------------------------------------------
# Kompilér modellen
# ------------------------------------------------------------
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# optimizer='adam': Adam anvender adaptive læringsrater til gradient descent.
# SparseCategoricalCrossentropy: En tabsfunktion til klassificering, når labels er heltal.
# from_logits=True: Angiver, at outputtet ikke er normaliseret (ingen softmax anvendes endnu).

# ------------------------------------------------------------
# Vis modelarkitektur
# ------------------------------------------------------------
model.summary()
# model.summary() udskriver en oversigt over modelarkitekturen og antallet af parametre for hvert lag.

# ------------------------------------------------------------
# Træn modellen
# ------------------------------------------------------------
epochs = 15  # Epoch: En fuld iteration over hele træningsdatasættet.
# Eksempel: Hvis vi har 100 batches og kører 15 epochs, trænes modellen over de 100 batches 15 gange.
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
# model.fit() træner modellen med træningsdata og evaluerer samtidig på validation data.
# history gemmer træningshistorikken, inklusiv nøjagtighed og tabsværdier for hver epoch.

# ------------------------------------------------------------
# Visualiser trænings- og valideringskurver
# ------------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')  
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')  # plt.legend() viser forklaring på graferne.
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# plt.show() viser de genererede plots i et vindue.

# ------------------------------------------------------------
# Test modellen med et eksternt billede
# ------------------------------------------------------------
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# Hent et billede af en solsikke fra nettet med tf.keras.utils.get_file()
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

# Indlæs billedet med den foruddefinerede størrelse (skal matche træningsdata)
img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
# load_img() indlæser billedet og skalerer det til target_size.

# Konverter billedet til en NumPy-array (til videre behandling i modellen)
img_array = tf.keras.utils.img_to_array(img)
# img_to_array() konverterer PIL-billedet til en array af pixelværdier.

# Udvid dimensionerne, så billedet ligner en batch med ét billede.
img_array = tf.expand_dims(img_array, 0)
# tf.expand_dims() tilføjer en batch-dimension, så modellen kan håndtere input som "batch".

# Få modelens forudsigelser for billedet
predictions = model.predict(img_array)
# model.predict() returnerer modelens output (logits) for input-billedet.

# Anvend softmax for at konvertere logits til normaliserede sandsynligheder.
score = tf.nn.softmax(predictions[0])
# tf.nn.softmax() sikrer, at summen af sandsynlighederne bliver 1.

# Udskriv resultaterne: hvilken klasse billedet mest sandsynligt tilhører, og med hvilken konfidens.
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
# np.argmax(score) returnerer indeks for den klasse med højeste sandsynlighed.
# np.max(score) returnerer den højeste sandsynlighed, som herefter ganges med 100 for at få procent.