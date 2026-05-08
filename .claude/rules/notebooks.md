---
paths:
  - "code/**/*.ipynb"
---

# Notebook Rules

- Each notebook has one clear purpose (training, evaluation, visualization)
- First cell: imports only. Second cell: config/hyperparameters as named constants
- Clear all outputs before committing (`Kernel > Restart & Clear Output`)
- Do not hardcode absolute paths — use `pathlib.Path` relative to notebook location
- Seed all randomness at the top: `torch.manual_seed(42)`, `np.random.seed(42)`
- Add a markdown cell before each major section explaining what it does
