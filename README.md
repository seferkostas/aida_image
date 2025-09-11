# Adversarial Deep Learning Experiment Framework in image classification

A modular framework for evaluating the robustness of deep learning models against adversarial attacks and the effectiveness of various defense methods on image data in classifation tasks.

## Features

- **Multi-Dataset Support**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- **Model Architectures**: CNN, ResNet
- **Attack Methods**: FGSM, BIM, PGD, DeepFool, C&W
- **Defense Techniques**: Adversarial Training FGSM, Adversarial Training PGD, feature squeezing, spatial smoothing, gaussian augmentation, jpeg compression

### Installation

```bash
# Clone the repository
git clone https://github.com/seferkostas/aida_image.git
cd aida_image

# Create virtual enviroment and activate
python -m venv .venv

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

```bash

# Create and run experiment (example)
 python main.py \
	 --datasets mnist cifar10 \
	 --models cnn resnet \
	 --attacks fgsm pgd \
	 --defenses none adversarial_training \
	 --epochs 10 \
	 --output-dir results/experiment-1
```

## Project Structure

```
aida-image/
├── config/                 # Experiment configurations
├── src/
│   ├── models/            # Model architectures
│   ├── attacks/           # Adversarial attack implementations
│   ├── defenses/          # Defense method implementations
│   ├── datasets/          # Data loading and preprocessing
│   ├── evaluation/        # Metrics and evaluation
│   └── utils/             # Utilities/visualizations
├── experiments/           # Experiment runner
├── results/              # Output directory
│   ├── plots/            # Created visualizations
│   ├── tables/           # Summary tables with results
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- **Adversarial Robustness Toolbox (ART)** for attack/defense implementations
- **TensorFlow** for the deep learning implementations