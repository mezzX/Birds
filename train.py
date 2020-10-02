import tensorflow as tf
import numpy as np
import random
import config
import tqdm
from tqdm import tqdm
from utils.data_utils import get_batch_data
from utils.losses import calc_losses
from model import AwesomeNet
random.seed(config.SEED)


def get_batch(file_list, batch_size=config.BATCH_SIZE):
    n = 0
    batch_files = []
    for i in range(len(file_list)):
        batch_files.append(file_list[i])
        n += 1
        if n >= batch_size:

            yield batch_files
            n = 0
            batch_files = []


def mk_list(file_list):
    img_files = []
    with open(file_list) as f:
        for line in f:
            img_files.append(line.strip())
    
    random.shuffle(img_files)

    return img_files


def train_val_split():
    train_files = mk_list(config.TRAIN_FILE)
    valid_files = train_files[:config.VALID_SIZE]
    train_files = train_files[config.VALID_SIZE:]
    num_train = len(train_files) // config.BATCH_SIZE
    num_valid = len(valid_files) // config.BATCH_SIZE

    return train_files, valid_files, num_train, num_valid


def model_step(model, imgs, lbls, train=True):
    with tf.GradientTape() as tape:
        outs = model(imgs, train)
        loss = calc_losses(outs, lbls)

        return loss, tape.gradient(loss, model.trainable_variables)


def train(save_dir, load_pt=None, ver=0):
    model = AwesomeNet()

    if load_pt != None:
        model.load_weights("./weights/{:s}/version_{:05d}".format(load_pt, ver))

    optimizer = tf.keras.optimizers.Adadelta(learning_rate = config.LEARNING_RATE)
    best_loss = float('inf')

    train_img, valid_img, num_train, num_valid = train_val_split()

    for e in range(config.EPOCHS + 1):
        batch_gen = get_batch(train_img)
        t_losses = []
        tqdm.write("Epoch: {:04d}, Training...".format(e))
        
        for step in tqdm(range(num_train)):
            batch_files = next(batch_gen)
            img_batch, label_batch = get_batch_data(batch_files)
            loss_train, grads = model_step(model, img_batch, label_batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            t_losses.append(loss_train.numpy())            

        batch_gen = get_batch(valid_img)
        v_losses = []
        tqdm.write("Epoch: {:04d}, Validating...".format(e))

        for step in tqdm(range(num_valid)):
            batch_files = next(batch_gen)
            img_batch, label_batch = get_batch_data(batch_files)
            loss_valid, _ = model_step(model, img_batch, label_batch, False)
            v_losses.append(loss_valid.numpy())

        valid_loss = np.mean(v_losses)
        train_loss = np.mean(t_losses)

        tqdm.write("Epoch: {:04d},  Training Loss: {:.6f},  Validation Loss: {:.6f}".format(e, train_loss, valid_loss))

        if valid_loss < best_loss:
            best_loss = valid_loss
            model.save_weights("./weights/{:s}/version_{:05d}".format(save_dir, ver))
            tqdm.write("Weights saved!")

        ver += 1
    
    print("Final Training Loss: {:.6f}, Final Validation Loss: {:.6f}".format(train_loss, valid_loss))


def main():
    train(config.SAVE_DIR)


if __name__ == "__main__":
    main()