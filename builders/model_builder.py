from model.FastICENet import FastICENet


def build_model(model_name, num_classes):
    if model_name == 'FastICENet':
        return FastICENet(classes=num_classes)
   