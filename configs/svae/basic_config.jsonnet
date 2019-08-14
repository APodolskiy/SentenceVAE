{
    model: {
        embed_dim: 300,
        latent_dim: 16,
        word_drop_p: 0.5,
        tie_weights: true,
        greedy: false,
        encoder: {
            input_size: $.model.embed_dim,
            hidden_size: 256,
            num_layers: 1,
            bidirectional: true,
            dropout: 0.4,
        },
        decoder: {
            input_size: $.model.embed_dim,
            hidden_size: 256,
            num_layers: 1,
            dropout: 0.4,
        },
        hidden_output_size: self.encoder.hidden_size * self.encoder.num_layers * (if self.encoder.bidirectional then 2 else 1),
    },
    training: {
        epochs: 20,
        batch_size: 128,
        test_batch_size: 128,
        optimizer: {
            lr: 1e-3,
            betas: [0.9, 0.999]
        },
    }
}