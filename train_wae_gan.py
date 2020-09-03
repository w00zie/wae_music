import tensorflow as tf

import os
import json
from datetime import datetime, timedelta
from time import time
from typing import Tuple

from utils.data import get_dataset
from utils.plot_utils import get_matrix_to_be_plotted
from utils.get_model import get_ae_disc
from utils.metrics import calc_prec_recall


class Train:
    def __init__(self, config: dict):
        # Experiment directory
        self.logdir = os.path.join("runs", datetime.now().strftime("wae_gan_%d_%m_%Y-%H:%M:%S"))
        self.writer = tf.summary.create_file_writer(self.logdir)
        # Logging the training config to Tensorboard
        with self.writer.as_default():
            tf.summary.text("Hyperparams", json.dumps(config), step=0)
        self.writer.flush()
        #os.mkdir(os.path.join(self.logdir,"img"))
        os.mkdir(os.path.join(self.logdir,"models"))
        os.mkdir(os.path.join(self.logdir,"models","encoder"))
        os.mkdir(os.path.join(self.logdir,"models","decoder"))
        os.mkdir(os.path.join(self.logdir,"models","discriminator"))
        with open(os.path.join(self.logdir,"config.json"), "w") as f:
            json.dump(config, f)

        self.h_dim = tf.constant(config["h_dim"], dtype=tf.int32)
        self.z_dim = tf.constant(config["z_dim"], dtype=tf.int32)
        self.epochs = config["epochs"]
        self.pre_epochs = config["pre_epochs"]
        self.batch_size = tf.constant(config["batch_size"], dtype=tf.int32)
        self.sigma_z = tf.constant(config["sigma_z"], dtype=tf.float32)
        self.lmbda = tf.constant(config["lambda"], dtype=tf.float32)
        self.c_weight = tf.constant(config["c_weight"], dtype=tf.float32)

        # Models --------------------------------------------------------------
        self.encoder, self.decoder, self.discriminator = get_ae_disc(config)

        # Optimizers ----------------------------------------------------------
        ae_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["ae_lr"],
            decay_steps=config["ae_dec_steps"],
            decay_rate=config["ae_dec_rate"])
        disc_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["d_lr"],
            decay_steps=config["d_dec_steps"],
            decay_rate=config["d_dec_rate"])

        self.enc_optim = tf.keras.optimizers.Adam(ae_scheduler)
        self.dec_optim = tf.keras.optimizers.Adam(ae_scheduler)
        self.disc_optim = tf.keras.optimizers.Adam(disc_scheduler)

        # Data -----------------------------------------------------------------
        tf.print("Loading training data...")
        self.train_dataset = get_dataset(filename=config["loc_train_array"],
                                         mmap=None,
                                         batch_size=tf.cast(self.batch_size, 
                                                            tf.int64))
        tf.print("Loading testing data...")
        self.test_dataset = get_dataset(filename=config["loc_test_array"],
                                        mmap=None,
                                        shuffle=False,
                                        test=True,
                                        batch_size=tf.cast(self.batch_size, 
                                                           tf.int64))
        tf.print("Done.")

        # Metric trackers -----------------------------------------------------
        ## Train --------------------------------------------------------------
        self.avg_d_train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_mse_train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_prec_train = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_rec_train = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_pre_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        ## Test ---------------------------------------------------------------
        self.avg_d_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_d_z_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_mse_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_pre_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_prec_test = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_rec_test = tf.keras.metrics.Mean(dtype=tf.float32)

    # Prior ===================================================================
    #@tf.function
    def sample_pz(self, batch_size: tf.Tensor) -> tf.Tensor:
        return tf.random.normal(shape=(batch_size, self.z_dim),
                                stddev=tf.sqrt(self.sigma_z))

    # Losses ==================================================================
    @tf.function
    def disc_loss(self, d_q_z: tf.Tensor, d_p_z: tf.Tensor) -> tf.Tensor:
        """Calculates the discriminator loss:
                    lambda * [ log D(pz) + log (1 - D(qz)) ]
        
        Parameters
        ----------
        d_q_z : tf.Tensor of shape `(batch_size, 1)`
            Tensor containing the estimated probabilities that the (fake) 
            samples in the batch are coming from the prior distribution.
        d_p_z : tf.Tensor of shape `(batch_size, 1)`
            Tensor containing the estimated probabilities that the (real) 
            samples in the batch are coming from the prior distribution.
        
        Returns
        -------
        A tf.Tensor containing the loss written above, averaged through
        `batch_size` samples.
        """
        loss_qz = tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(d_q_z), 
                                                      y_pred=d_q_z)
        loss_pz = tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(d_p_z), 
                                                      y_pred=d_p_z)
        return self.lmbda*(tf.reduce_mean(loss_qz) + tf.reduce_mean(loss_pz))

    @tf.function
    def ae_loss(self, batch: tf.Tensor, 
                x_hat: tf.Tensor, d_q_z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculates the WAE-GAN loss:
                    c(x, G(Q(x))) + lambda * log D(Q(x))
                    where c(x,y) = ||x - y||**2
        Parameters
        ----------
        batch: tf.Tensor of shape `(batch_size, 192, 84, 5)`
            The data from the original distribution.
        x_hat: tf.Tensor of shape `(batch_size, 192, 84, 5)`
            The data reconstruction operated by the AE.
        d_q_z: tf.Tensor of shape `(batch_size, 1)`
            Tensor containing the probabilities that the (fake) samples in 
            the batch are coming from the prior distribution.
        
        Returns
        -------
        ae_loss : tf.Tensor containing the loss written above, averaged through
        `batch_size` samples.
        mse : tf.Tensor
        """
        mse = self.c_weight * tf.reduce_mean(tf.reduce_sum(tf.math.square(batch - x_hat), axis=[1,2,3]))
        penalty = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(d_q_z), 
                                                                     y_pred=d_q_z))
        ae_loss = mse + self.lmbda * penalty
        return (ae_loss, mse)

    @tf.function
    def pre_train_loss(self, q_z: tf.Tensor) -> tf.Tensor:
        """A "pre-train" loss, as implemented here
        https://github.com/tolstikhin/wae/blob/master/wae.py#L130

            ||mu_pz - mu_qz||**2 + ||sigma_pz - sigma_qz||**2 

        where `mu_pz` and `mu_qz` are the means of latent vectors
        `pz` and `qz` respectively while `sigma_pz` and `sigma_qz` are
        their sample covariance. 

        Parameters
        ----------
        q_z: tf.Tensor of shape (batch_size, z_dim)
            The latent vector coming from the original distribution.

        Returns
        -------
        A tf.Tensor containing the loss written above, averaged through
        `batch_size` samples.
        """
        batch_size = tf.shape(q_z)[0]
        p_z = self.sample_pz(batch_size)
        mean_pz = tf.reduce_mean(p_z, axis=0, keepdims=True)
        mean_qz = tf.reduce_mean(q_z, axis=0, keepdims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
        div = tf.cast(batch_size - 1, tf.float32)
        cov_pz = tf.matmul(p_z - mean_pz,
                           p_z - mean_pz, transpose_a=True)
        cov_pz = tf.divide(cov_pz, div)
        cov_qz = tf.matmul(q_z - mean_qz,
                           q_z - mean_qz, transpose_a=True)
        cov_qz = tf.divide(cov_qz, div)
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        return mean_loss + cov_loss

    # Optimization steps ======================================================
    @tf.function
    def train_discriminator(self, batch: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(batch)[0]
        with tf.GradientTape() as tape:
            p_z = self.sample_pz(batch_size)
            d_p_z = self.discriminator(p_z, training=True)
            d_q_z = self.discriminator(self.encoder(batch, training=True), training=True)
            disc_loss = self.disc_loss(d_q_z=d_q_z, d_p_z=d_p_z)
        disc_grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optim.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        return disc_loss

    @tf.function
    def train_enc_dec(self, batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            q_z = self.encoder(batch, training=True)
            x_hat = self.decoder(q_z, training=True)
            d_q_z = self.discriminator(q_z, training=True)
            enc_dec_loss, mse = self.ae_loss(batch=batch, x_hat=x_hat, d_q_z=d_q_z)
        enc_grads = enc_tape.gradient(enc_dec_loss, self.encoder.trainable_variables)
        dec_grads = dec_tape.gradient(enc_dec_loss, self.decoder.trainable_variables)
        self.enc_optim.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))
        self.dec_optim.apply_gradients(zip(dec_grads, self.decoder.trainable_variables))
        return (x_hat, enc_dec_loss, mse)
    
    @tf.function
    def pre_train_enc_step(self, batch: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as enc_tape:
            q_z = self.encoder(batch, training=True)
            pre_loss = self.pre_train_loss(q_z)
        enc_grads = enc_tape.gradient(pre_loss, self.encoder.trainable_variables)
        self.enc_optim.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))
        return pre_loss

    # One epoch runs ==========================================================
    #@tf.function
    def train_step(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """This method executes one epoch of training on the `train_dataset`. 
        It calculates the average training losses during the epoch and returns them,
        along with other metrics of interest.
        """
        # Reset metrics
        self.avg_d_train_loss.reset_states()
        self.avg_enc_dec_train_loss.reset_states()
        self.avg_mse_train_loss.reset_states()
        self.avg_prec_train.reset_states()
        self.avg_rec_train.reset_states()
        # Run through an epoch of training
        for batch in self.train_dataset:
            x_hat, enc_dec_loss, mse = self.train_enc_dec(batch)
            disc_loss = self.train_discriminator(batch)
            prec, rec = calc_prec_recall(batch=batch, x_hat=x_hat)
            # Log metrics for every batch
            self.avg_d_train_loss(disc_loss)
            self.avg_enc_dec_train_loss(enc_dec_loss)
            self.avg_mse_train_loss(mse)
            self.avg_prec_train(prec)
            self.avg_rec_train(rec)
        return (
            self.avg_d_train_loss.result(), 
            self.avg_enc_dec_train_loss.result(),
            self.avg_mse_train_loss.result(),
            self.avg_prec_train.result(),
            self.avg_rec_train.result()
        )

    def pre_train_enc(self) -> None:
        """This method executes one epoch of pre-training on the `train_dataset`. 
        It calculates the average pre-train loss and logs it to Tensorboard.
        """
        for epoch in range(self.pre_epochs):
            self.avg_pre_loss.reset_states()
            for batch in self.train_dataset:
                self.avg_pre_loss(self.pre_train_enc_step(batch))
            mean_pre_loss = self.avg_pre_loss.result()
            tf.summary.scalar("Pre-Train/Matching Loss (mean+cov)", mean_pre_loss, step=epoch)
            tf.print("[PRETRAIN] Epoch {}/{}  -  [pre_loss] = {}]  -  {}".\
                format(epoch, self.pre_epochs-1, mean_pre_loss, 
                       datetime.now().strftime('%H:%M:%S')))

    #@tf.function
    def validation_step(self) -> Tuple[tf.Tensor, tf.Tensor, 
                                       tf.Tensor, tf.Tensor,
                                       tf.Tensor, tf.Tensor]:
        # Reset metrics
        self.avg_enc_dec_test_loss.reset_states()
        self.avg_mse_test_loss.reset_states()
        self.avg_d_test_loss.reset_states()
        self.avg_pre_test_loss.reset_states()
        self.avg_prec_test.reset_states()
        self.avg_rec_test.reset_states()
        # Run through an epoch of validation
        for batch in self.test_dataset:
            q_z = self.encoder(batch, training=False)
            x_hat = self.decoder(q_z, training=False)
            d_q_z = self.discriminator(q_z, training=False)
            precision, recall = calc_prec_recall(batch=batch, x_hat=x_hat)
            penalty = tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(d_q_z), 
                                                          y_pred=d_q_z)
            mse = self.c_weight * tf.reduce_sum(tf.math.square(batch - x_hat), axis=[1,2,3])
            enc_dec_loss = tf.reduce_mean(mse) + self.lmbda * tf.reduce_mean(penalty) 
            # Log metrics for every batch
            self.avg_d_test_loss(tf.reduce_mean(d_q_z))
            self.avg_mse_test_loss(tf.reduce_mean(mse))
            self.avg_pre_test_loss(tf.reduce_mean(self.pre_train_loss(q_z)))
            self.avg_enc_dec_test_loss(enc_dec_loss)
            self.avg_prec_test(precision)
            self.avg_rec_test(recall)
        return (
            self.avg_d_test_loss.result(), 
            self.avg_enc_dec_test_loss.result(), 
            self.avg_mse_test_loss.result(),
            self.avg_pre_test_loss.result(),
            self.avg_prec_test.result(),
            self.avg_rec_test.result()
        )

    # Plot utils ==============================================================
    def log_hist(self, step: int) -> None:
        for w in self.encoder.trainable_weights:
            tf.summary.histogram("Encoder/{}".format(w.name), w, step=step)
        for w in self.decoder.trainable_weights:
            tf.summary.histogram("Decoder/{}".format(w.name), w, step=step)
        for w in self.discriminator.trainable_weights:
            tf.summary.histogram("Discriminator/{}".format(w.name), w, step=step)

    def display_random_bar(self, epoch: int) -> None:
        """Samples 1 element from the latent space, decodes it into the 
        original image space and logs the (vertically-stacked) images to Tensorboard.
        The stacking is made "instrument"-wise, resulting in a matrix of shape (420, 192).
        """
        z = self.sample_pz(batch_size=tf.constant(1))
        x_hat = self.decoder(z, training=False)
        x_hat = tf.concat([x_hat[:,:,:,i] for i in range(5)], axis=2)
        x_hat = tf.expand_dims(x_hat, axis=-1)
        x_hat = tf.transpose(x_hat)
        tf.summary.image('Random Draw', x_hat, epoch)
        
    def display_reconstruction(self, epoch: int) -> None:
        """Samples 1 element from the original distribution, encodes it into the 
        latent space and then decodes it back into the original image space. 
        Two (vertically-stacked) images are then logged to Tensorboard.
        The stacking is made "instrument"-wise, resulting in two matrices of shape (420, 192).
        """
        for batch in self.test_dataset.take(1):
            x_hat = self.decoder(self.encoder(batch, training=False), training=False)
            x_hat = tf.concat([x_hat[0,:,:,i] for i in range(5)], axis=1)
            x_hat = tf.transpose(x_hat)
            x_hat = tf.expand_dims(x_hat, axis=0)
            x_hat = tf.expand_dims(x_hat, axis=-1)
            batch = tf.concat([batch[0,:,:,i] for i in range(5)], axis=1)
            batch = tf.transpose(batch)
            batch = tf.expand_dims(batch, axis=0)
            batch = tf.expand_dims(batch, axis=-1)
            tf.summary.image('Reconstruction/True', batch, epoch)
            tf.summary.image('Reconstruction/Recon', x_hat, epoch)
    
    def display_encoder_features(self, epoch: int) -> None:
        """This method logs to Tensorboard the (stacked) images
        of the various filters coming from the activations in the encoder network.
        It lets you take a look at what's going on in the hidden layers of the CNN.
        """
        enc_model = self.get_activation_model(self.encoder)
        for batch in self.test_dataset.take(1):
            activations = enc_model(batch)
            for i, act in enumerate(activations):
                mat = get_matrix_to_be_plotted(act[0], num_cols=16)
                mat = tf.expand_dims(mat, axis=0)
                mat = tf.expand_dims(mat, axis=-1)
                tf.summary.image("Reconstruction Features/activation-{}".format(i), 
                                 mat, 
                                 step=epoch)
    
    def display_decoder_features(self, epoch: int) -> None:
        """This method logs to Tensorboard the (stacked) images
        of the various filters coming from the activations in the decoder network.
        It lets you take a look at what's going on in the hidden layers of the CNN.
        """
        dec_model = self.get_activation_model(self.decoder)
        pz = self.sample_pz(batch_size=tf.constant(1))
        activations = dec_model(pz)
        for i, act in enumerate(activations):
            mat = get_matrix_to_be_plotted(act[0], num_cols=16)
            mat = tf.expand_dims(mat, axis=0)
            mat = tf.expand_dims(mat, axis=-1)
            tf.summary.image("Construction Features/activation-{}".format(i), 
                             mat, 
                             step=epoch)

    # General utils ===========================================================
    #@tf.function
    def calc_discriminator_latent(self) -> tf.Tensor:
        """This method calculates the mean value coming from the discrimination 
        of 32 batches of `batch_size` samples from the prior distribution. 
        """
        self.avg_d_z_loss.reset_states()
        for _ in tf.range(32):
            pz = self.sample_pz(self.batch_size)
            d_p_z = self.discriminator(pz, training=False)
            self.avg_d_z_loss(tf.reduce_mean(d_p_z))
        return self.avg_d_z_loss.result()
        
    def save_weights(self) -> None:
        self.encoder.save_weights(os.path.join(self.logdir,"models","encoder","encoder"))
        self.decoder.save_weights(os.path.join(self.logdir,"models","decoder","decoder"))
        self.discriminator.save_weights(os.path.join(self.logdir,"models",
                                                     "discriminator","discriminator"))

    def get_activation_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """This method is used to retrieve the activations of the convolutional
        layers in a model. These activations are marked with an `act` in their names, so
        they are more easily accessible.
        """
        outputs = [l.output for l in model.layers if 'act' in l.name]
        return tf.keras.Model(inputs=model.input, outputs=outputs)

    # Training procedure ======================================================
    def train(self) -> None:
        with self.writer.as_default():
            if self.pre_epochs > 0:
                self.pre_train_enc()    
            start = time()
            for epoch in range(self.epochs):
                # Train for one epoch
                d_loss, e_d_loss, mse_train, prec_train, rec_train = self.train_step()
                # Display a few bars (as images) in Tensorboard and log weights
                if epoch % 5 == 0:
                    self.display_random_bar(epoch)
                    self.display_reconstruction(epoch)
                    #self.save_weights()
                if epoch % 10 == 0:
                    self.save_weights()
                #    self.display_encoder_features(epoch)
                #    self.display_decoder_features(epoch)
                # Validate on the test set
                d_g_z, test_loss, mse_test, pre_test_loss, prec_test, rec_test = \
                    self.validation_step()
                # Calculate D(z)
                d_z = self.calc_discriminator_latent()
                # Logging
                tf.summary.scalar("Train/Discriminator Loss", 
                                  d_loss, step=epoch)
                tf.summary.scalar("Train/Encoder-Decoder Loss", 
                                  e_d_loss, step=epoch)
                tf.summary.scalar("Train/MSE", mse_train, step=epoch)
                tf.summary.scalar("Train/Precision", prec_train, step=epoch)
                tf.summary.scalar("Train/Recall", rec_train, step=epoch)
                tf.summary.scalar("Learning Rates/Encoder", 
                                  self.enc_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Learning Rates/Decoder", 
                                  self.dec_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Learning Rates/Discriminator", 
                                  self.disc_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Test/D(G(x))", d_g_z, step=epoch)
                tf.summary.scalar("Test/D(z)", d_z, step=epoch)
                tf.summary.scalar("Test/Encoder-Decoder Loss", 
                                  test_loss, step=epoch)
                tf.summary.scalar("Test/MSE", mse_test, step=epoch)
                tf.summary.scalar("Test/Pre-Loss", pre_test_loss, step=epoch)
                tf.summary.scalar("Test/Precision", prec_test, step=epoch)
                tf.summary.scalar("Test/Recall", rec_test, step=epoch)
                self.log_hist(epoch)

                tf.print("Epoch {}/{}  -  [d_loss = {}]  -  [ae_loss = {}]  -  {}".\
                    format(epoch, self.epochs-1, d_loss.numpy(), 
                           e_d_loss.numpy(), datetime.now().strftime('%H:%M:%S')))
       
        print(f"Training took {timedelta(seconds=time()-start)}")

        self.save_weights()
        self.writer.flush()

if __name__ == "__main__":

    from configs.wae_gan_config import config

    # Logging only (W)arnings and (E)rrors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    train = Train(config)
    train.train()
