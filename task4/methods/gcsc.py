from task4.models.resnet_cifar import (
    CIFARResNet18,
)


def build_gcsc_model():
    return CIFARResNet18(
        number_of_outputs=10
    )
