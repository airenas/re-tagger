import tensorflow as tf

from egs.bilstm_crf.local.format_data import ending


def make_train_dataset(data):
    def create_data_generator(dataset):
        def data_generator():
            for item in dataset:
                # print(len(item['tokens']))
                yield item['tokens'], [ending(w) for w in item['tokens']], item['tags']

        return data_generator

    data_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.string)
    )

    return tf.data.Dataset.from_generator(
        create_data_generator(data),
        output_signature=data_signature
    )
