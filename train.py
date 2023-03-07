class Trainer():
    def train(self):
        pass

    def loss_fn(self, a_hat_dist, a, attention_mask, entropy_reg):
        # a_hat is a SquashedNormal Distribution
        log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

        entropy = a_hat_dist.entropy().mean()
        loss = -(log_likelihood + entropy_reg * entropy)

        return (
            loss,
            -log_likelihood,
            entropy,
        )
