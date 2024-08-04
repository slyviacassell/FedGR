import torchvision

mnist_train = torchvision.datasets.MNIST(root="rawdata/mnist", train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root="rawdata/mnist", train=False, download=True)

cifar10_train = torchvision.datasets.CIFAR10(root="rawdata/cifar10", train=True, download=True)
cifar10_test = torchvision.datasets.CIFAR10(root="rawdata/cifar10", train=False, download=True)

cifar100_train = torchvision.datasets.CIFAR100(root="rawdata/cifar100", train=True, download=True)
cifar100_test = torchvision.datasets.CIFAR100(root="rawdata/cifar100", train=False, download=True)

svhn_train = torchvision.datasets.SVHN(root="rawdata/svhn", split="train", download=True)
svhn_test = torchvision.datasets.SVHN(root="rawdata/svhn", split="test", download=True)
