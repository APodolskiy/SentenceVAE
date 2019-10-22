function (
    // model hyperparams
    latent_dim = 32,
    embed_dim = 512,
    hidden_dim = 1024,
    p_drop = 0.5,
    p_word_drop = 0.3,
    embedding_drop_p = 0.5,
    out_drop_p = 0.5,
    tie_weights = false,
    encoder_num_layers = 1,
    decoder_num_layers = 1,
    // training hyperparams
    epochs = 40,
    annealing_type = 'linear',
    annealing_steps = 10000,
    learning_rate = 1e-3,
    beta1 = 0.9,
    cuda_device = 0
)
{
    model: {
        embed_dim: embed_dim,
        embedding_drop_p: embedding_drop_p,
        out_drop_p: out_drop_p,
        latent_dim: latent_dim,
        word_drop_p: p_word_drop,
        tie_weights: tie_weights,
        greedy: false,
        encoder: {
            rnn_type: 'lstm',
            input_size: $.model.embed_dim,
            hidden_size: hidden_dim,
            num_layers: encoder_num_layers,
            bidirectional: true,
            dropout: p_drop,
        },
        decoder: {
            rnn_type: 'lstm',
            input_size: $.model.embed_dim,
            hidden_size: hidden_dim,
            num_layers: decoder_num_layers,
            dropout: p_drop,
        },
        hidden_output_size: self.encoder.hidden_size * self.encoder.num_layers * (if self.encoder.bidirectional then 2 else 1),
        annealing:
        // Two types of annealing functions exists: logistic and linear
        {
            type: annealing_type,
            max_value: 1.0,
            steps: annealing_steps,
            warm_up_steps: 0,
            start_value: 0.01,
        } +
        if annealing_type == 'logistic' then{
            fast: false,
            eps: 1e-3
        }
        else {}
    },
    training: {
        epochs: epochs,
        batch_size: 32,
        test_batch_size: 64,
        optimizer: {
            lr: learning_rate,
            betas: [beta1, 0.999]
        },
        lr_scheduler: {
            start_epoch: 30,
            decay_interval: 2,
            lr_multiplier: 0.5
        },
        cuda_device: cuda_device,
    },
    sampling: {
        temperature: 0.8,
        max_len: 150,
    },
    eval_on_test: true,
    dataset: {
        name: "YelpReview",
    },
}