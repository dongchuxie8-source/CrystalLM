# Contributing to CrystalLM

Thank you for your interest in contributing to CrystalLM! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/dongchuxie8-source/CrystalLM.git
   cd CrystalLM
   ```
3. Create a virtual environment:
   ```bash
   conda create -n crystallm python=3.9
   conda activate crystallm
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

We use the following tools for code formatting and linting:
- **black** for code formatting
- **flake8** for linting
- **isort** for import sorting

Before submitting a PR, please run:
```bash
black crystallm/ scripts/ tests/
flake8 crystallm/ scripts/ tests/
isort crystallm/ scripts/ tests/
```

### Testing

Run tests with pytest:
```bash
pytest tests/ -v
```

For coverage report:
```bash
pytest tests/ --cov=crystallm --cov-report=html
```

### Commit Messages

Please use clear and descriptive commit messages:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters

Examples:
- `Add HSSR validation for zeolite structures`
- `Fix tokenization issue in T2S translation`
- `Update documentation for training scripts`

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Reference to related issues (if any)
   - Test results

5. Wait for review and address any feedback

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Feel free to open an issue for any questions about contributing.
