import os
from glob import glob
from data import imgproc
from autocorrect import Speller
from network.model import MyModel
from data.tokenizer import Tokenizer
from data import data_preprocessor as pp 
from config.config import target_path, target_image_size

def predict(image_name): 
    output_image_path = os.path.join("api", "temp")
    input_image_path = os.path.join(output_image_path, image_name)

    tokenizer = Tokenizer()

    model = MyModel(vocab_size=tokenizer.vocab_size,
                    beam_width=20,
                    stop_tolerance=15,
                    reduce_tolerance=10)
    model.compile(learning_rate=0.001)
    model.load_checkpoint(target=target_path)

    imgproc.__execute__(input_image_path, output_image_path)

    text = []
    confidence = []

    image_lines = sorted(glob(os.path.join(output_image_path, image_name.split('.')[0], "lines", "*.png")))

    for img in image_lines:
        img = pp.preprocess_image(img, target_image_size, predict=True)
        img = pp.normalization([img])

        predicts, probabilities = model.predict(img, ctc_decode=True)

        predicts = tokenizer.sequences_to_texts(predicts)
        confidence.append(f"{predicts[0]} ==> {probabilities[0]}")
        text.append(Speller("en").autocorrect_sentence(predicts[0][0]))

    return "\n".join(text)