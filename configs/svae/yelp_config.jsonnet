local annealing_type = 'linear';

{
    model: {
        model_type: "svae",
        embed_dim: 512,
        embedding_drop_p: 0.5,
        latent_dim: 32,
        word_drop_p: 0.3,
        tie_weights: false,
        greedy: false,
        encoder: {
            rnn_type: 'lstm',
            input_size: $.model.embed_dim,
            hidden_size: 1024,
            num_layers: 1,
            bidirectional: true,
            dropout: 0.5,
        },
        decoder: {
            rnn_type: 'lstm',
            input_size: $.model.embed_dim,
            hidden_size: 1024,
            num_layers: 1,
            dropout: 0.5,
        },
        hidden_output_size: self.encoder.hidden_size * self.encoder.num_layers * (if self.encoder.bidirectional then 2 else 1),
        annealing:
        // Two types of annealing functions exists: logistic and linear
        {
            type: annealing_type,
            max_value: 1.0,
            steps: 10000,
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
        epochs: 40,
        batch_size: 32,
        test_batch_size: 64,
        optimizer: {
            lr: 1e-3,
            betas: [0.5, 0.999]
        },
        lr_scheduler: {
            start_epoch: 30,
            decay_interval: 2,
            lr_multiplier: 0.5
        },
        cuda_deivce: 0,
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