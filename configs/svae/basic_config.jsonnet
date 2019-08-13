{
    model: {
        embed_dim: 300,
        latent_dim: 16,
        word_drop_p: 0.2,
        tie_weights: false,
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
        hidden_output_size: self.encoder.hidden_size * (if self.encoder.bidirectional then 2 else 1),
    },
    training: {
        epochs: 10,
        batch_size: 32,
        test_batch_size: 128,
        optimizer: {
            lr: 1e-3,
            betas: [0.9, 0.999]
        },
    }
}