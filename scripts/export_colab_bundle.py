"""Export helper for Colab bundle."""

from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    print("Files and folders to copy or push for Colab:")
    skip_names = {
        "__pycache__", ".pytest_cache",
        "build", "dist", "outputs", "checkpoints", "runs", "wandb",
    }
    for p in sorted(repo_root.rglob("*")):
        rel = p.relative_to(repo_root)
        # Skip hidden and generated files
        if any(part.startswith(".") or part.startswith("__") for part in rel.parts):
            continue
        if any(part in skip_names or part.endswith(".egg-info") for part in rel.parts):
            continue
        if p.is_dir():
            print(f"  DIR  {rel}/")
        else:
            print(f"  FILE {rel}")


if __name__ == "__main__":
    main()
