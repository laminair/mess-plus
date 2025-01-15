from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_size, num_labels, config):
        super().__init__()

        layer_dims: list = config["classifier_model"]["classifier_layer_hidden_dims"]
        layer_dims.insert(0, hidden_size)
        layer_dims.append(num_labels)

        self.mlp = nn.ModuleList([
            nn.Linear(hidden_size, layer_dims[1]),
            nn.ReLU()
        ])

        for idx, layer_dim in enumerate(layer_dims[2:]):
            linear_layers = [i for i in self.mlp if hasattr(i, "out_features")]
            self.mlp.append(
                nn.Linear(linear_layers[idx].out_features, layer_dim),
            )

            if not layer_dim == num_labels:
                self.mlp.append(
                    nn.ReLU()
                )

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)


def make_mlp(base_model: nn.Module, config):
    mlp = MLP(
        hidden_size=base_model.config.hidden_size,
        num_labels=base_model.config.num_labels,
        config=config
    ).to(base_model.device)

    return mlp


if __name__ == "__main__":

    cfg = {
        "classifier_model": {"classifier_layer_hidden_dims": [128, 256, 128, 64]}
    }

    mlp = make_mlp(base_model=nn.Module(), config=cfg)
    print(mlp)
