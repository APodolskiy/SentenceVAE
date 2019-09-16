local annealing_type = 'logistic';

{
    model: {
        embed_dim: 300,
        embedding_drop_p: 0.5,
        latent_dim: 16,
        word_drop_p: 0.3,
        tie_weights: false,
        greedy: false,
        encoder: {
            rnn_type: 'lstm',
            input_size: $.model.embed_dim,
            hidden_size: 256,
            num_layers: 1,
            bidirectional: true,
            dropout: 0.4,
        },
        decoder: {
            rnn_type: 'lstm',
            input_size: $.model.embed_dim,
            hidden_size: 256,
            num_layers: 1,
            dropout: 0.4,
        },
        hidden_output_size: self.encoder.hidden_size * self.encoder.num_layers * (if self.encoder.bidirectional then 2 else 1),
        annealing:
        // Two types of annealing functions exists: logistic and linear
        {
            type: annealing_type,
            max_value: 1.0,
            steps: 8000,
            warm_up_steps: 0,
        } +
        if annealing_type == 'logistic' then{
            fast: false,
            eps: 1e-3
        }
        else {}
    },
    training: {
        epochs: 50,
        batch_size: 128,
        test_batch_size: 128,
        optimizer: {
            lr: 1e-3,
            betas: [0.9, 0.999]
        },
    },
    sampling: {
        temperature: [0.1, 0.8],
    },
    eval_on_test: true,
    dataset: {
        name: "PTB"
    },
}