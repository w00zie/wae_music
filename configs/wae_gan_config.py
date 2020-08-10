config = {"h_dim": 16,
          "z_dim": 16,
          "epochs": 1200,
          "pre_epochs": 0,
          "batch_size": 64,
          "net_type": "new_alt",
          "conv_kernel_size": [6,6],
          "kernel_init": "TruncatedNormal",
          "disc_units": 512,
          "sigma_z": 1.,
          "d_lr": 1e-4,
          "d_dec_steps": 12800,
          "d_dec_rate": 0.9,
          "ae_lr": 1e-3,
          "ae_dec_steps": 9600,
          "ae_dec_rate": 0.8,
          "loc_train_array": "/home/giovanni_bindi41/src/data/4096_samples.npy",
          "loc_test_array": "/home/giovanni_bindi41/src/data/1024_samples.npy",
          "lambda": 100.}