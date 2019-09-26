import tensorflow as tf
from keras.callbacks import Callback
from keras.callbacks import TensorBoard

import io
from PIL import Image



def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    height, width, channel = tensor.shape
    print(tensor)
    image = Image.fromarray(tensor.astype('uint8'))  # TODO: maybe float ?

    output = io.BytesIO()
    image.save(output, format='JPEG')
    image_string = output.getvalue()
    output.close()

    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)

def weight_visualization(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    height, width, channel = tensor.shape
    print(tensor)
    image = Image.fromarray(tensor.astype('float32'))  # TODO: maybe float ?

    output = io.BytesIO()
    image.save(output, format='JPEG')
    image_string = output.getvalue()
    output.close()

    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardImage(Callback):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img_input = self.validation_data[0][0]  # X_train
        img_valid = self.validation_data[1][0]  # Y_train

        print(self.validation_data[0].shape)  # (8, 128, 128, 3)
        print(self.validation_data[1].shape)  # (8, 512, 512, 3)

        image = make_image(img_input)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        image = make_image(img_valid)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return


tbi_callback = TensorBoardImage('Image test')