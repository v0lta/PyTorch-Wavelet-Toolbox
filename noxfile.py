"""This module implements our CI function calls."""
import nox


@nox.session(name="test")
def run_test(session):
    """Run pytest."""
    session.install(".")
    session.install("pytest")
    session.run("pytest")


@nox.session(name="fast-test")
def run_test_fast(session):
    """Run pytest."""
    session.install(".")
    session.install("pytest")
    session.run("pytest", "-m", "not slow")


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("flake8")
    session.install(
        "flake8-colors",
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.install("flake8-bandit==2.1.2", "bandit==1.7.2")
    session.run("flake8", "src", "tests", "noxfile.py")


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install(".")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--strict",
        "--no-warn-return-any",
        "--implicit-reexport",
        "--allow-untyped-calls",
        "src",
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "tests", "noxfile.py")
    session.run("black", "src", "tests", "noxfile.py")


@nox.session(name="coverage")
def check_coverage(session):
    """Check test coverage and generate a html report."""
    session.install(".")
    session.install("pytest")
    session.install("coverage")
    try:
        session.run("coverage", "run", "-m", "pytest")
    finally:
        session.run("coverage", "html")


@nox.session(name="coverage-clean")
def clean_coverage(session):
    """Remove the code coverage website."""
    session.run("rm", "-r", "htmlcov", external=True)



@nox.session(name="build")
def build(session):
    session.install("wheel")
    session.install("setuptools")
    session.run("python", "setup.py", "-q", "sdist", "bdist_wheel")

@nox.session(name="finish")
def finish(session):
    session.install("bump2version")
    session.install("twine")
    session.run("bumpversion", "release", external=True)
    build(session)
    session.run("twine", "upload", "--skip-existing", "dist/*", external=True)
    session.run("git", "push", external=True)
    session.run("bumpversion", "patch", external=True)
    session.run("git", "push", external=True)
