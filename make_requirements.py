import subprocess

packages = [
    "fastapi",
    "pydantic",
    "uvicorn",
    "pandas",
    "joblib",
    "scikit-learn",
]


def get_version(pkg: str) -> str:
    result = subprocess.run(
        ["python", "-m", "pip", "show", pkg],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"Version not found for {pkg}")


def main():
    with open("requirements.txt", "w", encoding="utf-8") as f:
        for pkg in packages:
            version = get_version(pkg)
            f.write(f"{pkg}=={version}\n")


if __name__ == "__main__":
    main()
