from torch import nn

def create_dense_architecture(
    input_dimension,
    configuration_hidden_layers,
    output_dimension,
    activation_function_class,
    final_activation_function_class=None,
    layer_normalization=False,
    dropout=False,
    bias_final_layer=True,
    flatten_input=False,
    output_shape=None  # Nouvel argument ajouté
):
    layers = []

    # Ajouter une couche Flatten si flatten_input est True
    if flatten_input:
        layers.append(nn.Flatten())

    if len(configuration_hidden_layers) != 0:
        layers.append(nn.Linear(input_dimension, configuration_hidden_layers[0]))
        layers.append(activation_function_class())

        for i in range(len(configuration_hidden_layers) - 1):
            layers.append(nn.Linear(configuration_hidden_layers[i], configuration_hidden_layers[i + 1]))
            layers.append(activation_function_class())
            if layer_normalization:
                layers.append(nn.LayerNorm(configuration_hidden_layers[i + 1]))
            if dropout:
                layers.append(nn.Dropout(p=0.2))

        layers.append(nn.Linear(configuration_hidden_layers[-1], output_dimension, bias=bias_final_layer))

        # Ajouter la fonction d'activation finale si elle est spécifiée
        if final_activation_function_class is not None:
            layers.append(final_activation_function_class())

    else:
        # Pas de hidden layers
        layers.append(nn.Linear(input_dimension, output_dimension, bias=bias_final_layer))
        if final_activation_function_class is not None:
            layers.append(final_activation_function_class())

    # Reshape de la sortie si output_shape est fourni
    if output_shape is not None:
        layers.append(nn.Unflatten(1, output_shape))

    return nn.Sequential(*layers)
