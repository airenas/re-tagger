from tqdm import tqdm


def predict(model, file, data, in_func, out_func):
    with tqdm(total=len(data), desc="predicting") as pbar:
        for item in data:
            pbar.update(1)
            tokens = item['tokens']
            inputs = in_func(tokens)
            outputs = model(inputs)
            prediction = out_func(outputs[0])
            for i, w in enumerate(tokens):
                print("{}\t{}".format(w, str(prediction[i].numpy(), "utf-8")), file=file)


def predict_ds(file, data, predictions):
    with tqdm(total=len(data), desc="saving") as pbar:
        for item in data:
            pbar.update(1)
            tokens = item['tokens']
            prediction = next(predictions)
            for i, w in enumerate(tokens):
                print("{}\t{}".format(w, str(prediction[i].numpy(), "utf-8")), file=file)
